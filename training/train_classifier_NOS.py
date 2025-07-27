### Modified from https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/main/classifier.py ###
import itertools
import typing
import os

import hydra.utils
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
import transformers

#import dataloader
#import models.dit
import models.pretraining.model.mdlm.noise_schedule as noise_schedule


class MicroAveragingMetric(torchmetrics.Metric):
  """Micro-averaging metric.

    Adapted from https://github.com/HazyResearch/hyena-dna/blob/main/src/tasks/metrics.py#L12
  """

  def __init__(self, class_idx: typing.Optional[int] = 1,
               dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.class_idx = torch.tensor(class_idx) \
      if class_idx is not None else None
    self.add_state("numerator", default=torch.tensor(0.0),
                   dist_reduce_fx="sum")
    self.add_state("denominator", default=torch.tensor(0.0),
                   dist_reduce_fx="sum")

  def _update(
      self, numerator, denominator, preds, y) -> tuple:
    raise NotImplementedError

  def update(self, logits: torch.Tensor, y: torch.Tensor):
    # update metric states
    preds = torch.argmax(logits, dim=-1)
    y = y.view(-1)
    assert preds.shape == y.shape, \
      f"preds shape {preds.shape} != y shape {y.shape}"
    self.numerator, self.denominator = self._update(
      self.numerator, self.denominator, preds, y)

  def compute(self):
    # compute final result
    value = self.numerator.float() / self.denominator \
      if self.denominator.item() > 0. else torch.tensor(0.0)
    return value

  def reset(self):
    self.numerator = torch.tensor(0.0).to(self.device)
    self.denominator = torch.tensor(0.0).to(self.device)


class CrossEntropy(MicroAveragingMetric):
  """Calculates cross-entropy loss."""
  def _update(
      self, numerator, denominator, logits, y) -> tuple:
    with torch.no_grad():
      numerator += F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
        ignore_index=-100,
        reduction='sum')
      denominator += y.numel()
    return numerator, denominator

  # Overrides parent class to use logits and not (argmax) preds
  def update(self, logits: torch.Tensor, y: torch.Tensor):
    y = y.view(-1)
    self.numerator, self.denominator = self._update(
      self.numerator, self.denominator, logits, y)


class Accuracy(MicroAveragingMetric):
  """Calculates accuracy.

    Can be used to calculate accuracy per class.
    Copied from:
      https://github.com/HazyResearch/hyena-dna/blob/main/src/tasks/metrics.py
  """

  def _update(
      self, numerator, denominator, preds, y) -> tuple:
    if self.class_idx is None:
      numerator += (preds == y).sum()
      denominator += y.numel()
    else:
      class_idx = self.class_idx
      relevant_idxs = (y == class_idx)
      numerator += (preds[relevant_idxs] == class_idx).sum()
      denominator += relevant_idxs.sum()
      relevant_idxs = (y != class_idx)
      numerator += (preds[relevant_idxs] != class_idx).sum()
      denominator += relevant_idxs.sum()
    return numerator, denominator


class Precision(MicroAveragingMetric):
  """Calculates precision.

    Can be used to calculate precision per class.
    Adapted from:
      https://github.com/HazyResearch/hyena-dna/blob/main/src/tasks/metrics.py
  """

  def _update(self, numerator, denominator, preds, y) -> tuple:
    class_idx = self.class_idx
    relevant_idxs = (preds == class_idx)
    numerator += (y[relevant_idxs] == class_idx).sum()
    denominator += relevant_idxs.sum()
    return numerator, denominator


class Recall(MicroAveragingMetric):
  """Calculate recall.

    Can be used to calculate recall per class.
    Adapted from:
      https://github.com/HazyResearch/hyena-dna/blob/main/src/tasks/metrics.py
  """

  def _update(self, numerator, denominator, preds, y) -> tuple:
    class_idx = self.class_idx
    relevant_idxs = (y == class_idx)
    numerator += (preds[relevant_idxs] == class_idx).sum()
    denominator += relevant_idxs.sum()
    return numerator, denominator


class Classifier(L.LightningModule):
  def __init__(
      self,
      config,
      tokenizer,
      pretrained_backbone: typing.Optional[torch.nn.Module] = None
      ):
    super().__init__()
    self.save_hyperparameters(ignore=['pretrained_backbone'])
    self.config = config

    # This param indicates whether this model will be used
    #  for guidance (False) or only evaluation (True).
    self.is_eval_classifier = getattr(
      config, 'is_eval_classifier', False)

    self.tokenizer = tokenizer
    #self.vocab_size = tokenizer.vocab_size
    #self.tokenizer = hydra.utils.instantiate(tokenizer, sequences=True)
    self.vocab_size = len(self.tokenizer.alphabet)
    # self.antithetic_sampling = config.train.antithetic_sampling
    # self.importance_sampling = config.train.importance_sampling
    # self.change_of_variables = config.train.change_of_variables
    # if (not hasattr(self.tokenizer, 'mask_token')
    #     or self.tokenizer.mask_token is None):
    #   self.mask_index = self.vocab_size
    #   self.vocab_size += 1
    # else:
    #   self.mask_index = self.tokenizer.mask_token_id

    if config.classifier_backbone == 'dit':
      self.classifier_model = hydra.utils.instantiate(config.model, vocab_size=self.vocab_size)
    else:
      raise NotImplementedError(
        f"Classifier backbone "
        f"{self.config.classifier_backbone} not "
        f"implemented.")
    
    if pretrained_backbone is not None:  # For PPLM / NOS
      self.classifier_model.load_pretrained_encoder(
        pretrained_backbone)
    # Metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'cross_entropy': CrossEntropy(),
      'accuracy': Accuracy(class_idx=None),
    })
    # if config.data.num_classes > 2:
    #   for c in range(config.data.num_classes):
    #     metrics.add_metrics(
    #       {f"accuracy_class{c}": Accuracy(class_idx=c),
    #        f"precision_class{c}": Precision(class_idx=c),
    #        f"recall_class{c}": Recall(class_idx=c)})
    # else:
    metrics.add_metrics(
      {'precision': Precision(class_idx=1),
        'recall': Recall(class_idx=1)})
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')

    #self.T = config.T
    # self.noise = noise_schedule.get_noise(config,
    #                                       dtype=self.dtype)
    #self.sampling_eps = config.training.sampling_eps
    #self.lr = config.optim.lr
    self.time_conditioning = config.time_conditioning
    self.fast_forward_epochs = None
    self.fast_forward_batches = None

  def forward(self, x, sigma=None, x_emb=None, attention_mask=None):
    """Returns logits.

      x_emb can be provided during PPLM / NoS-style guidance
      (see: https://arxiv.org/abs/2305.20009).
    """
    if self.is_eval_classifier:
      logits = self.classifier_model(x)
      if hasattr(logits, 'logits'):
        logits = logits.logits
    else:
      sigma = self._process_sigma(sigma) if sigma is not None else sigma
      with torch.amp.autocast("cuda", dtype=torch.float32):
        logits = self.classifier_model(x, sigma, x_emb=x_emb, attention_mask=attention_mask)
    return logits

  def get_log_probs(self, x, sigma, x_emb=None):
    """Returns log probabilities.
      Use for CBG-style guidance.
    """
    if self.is_eval_classifier:
      raise NotImplementedError(
        '`get_log_prob` not implemented for classifiers '
        'that are meant to be used for evaluation purposes '
        'only.')
    with torch.cuda.amp.autocast(dtype=torch.float32):
      return torch.nn.functional.log_softmax(
        self.forward(x, sigma, x_emb=x_emb), dim=-1)

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True,
             prog_bar=True)
    self.log(name='lr',
             value=
             self.trainer.optimizers[0].param_groups[0][
               'lr'],
             on_step=True,
             on_epoch=False,
             sync_dist=True,
             prog_bar=True, logger=False)
    return loss

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.Adam(self.classifier_model.output_layer.parameters(), lr=0.1)
      # lr=self.config.optim.lr,
      # lr=0.1,
      # betas=(self.config.optim.beta1,
      #        self.config.optim.beta2),
      # eps=self.config.optim.eps,
      # weight_decay=self.config.optim.weight_decay)
    # optimizer = torch.optim.AdamW(
    #   itertools.chain(self.classifier_model.parameters(),
    #                   self.noise.parameters()),
    #   lr=self.config.optim.lr,
    #   betas=(self.config.optim.beta1,
    #          self.config.optim.beta2),
    #   eps=self.config.optim.eps,
    #   weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  def _q_xt(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      move_chance: float torch.Tensor with shape
        (batch_size, 1).
    """
    move_indices = torch.rand(
      *x.shape, device=x.device) < move_chance
    if self.config.diffusion == 'absorbing_state':
      return torch.where(move_indices, self.mask_index, x)
    if self.config.diffusion == 'uniform':
      uniform_tensor = torch.randint(
        0, self.vocab_size, x.shape, device=x.device)
      return torch.where(move_indices, uniform_tensor, x)
    raise NotImplementedError(
        f'Diffusion type {self.config.diffusion} not '
        'implemented.')

  def _process_sigma(self, sigma):
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

def train_step(classifier, model, optimizer, criterion, x, t, y, project_fn=None, time_conditioned=True):
    '''
    classifier: the classifier model
    model: the discrete diffusion model
    '''
    optimizer.zero_grad()
    xt = model.q_sample(x, t) if time_conditioned else x
    xt = project_fn(xt) if project_fn is not None else xt
    y_pred = classifier(xt, t).squeeze()
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_classifier(classifier, model, dataloader, train_config, save_dir=None, project_fn=None, ensemble_idx=0, time_conditioned=True):
  ### Adapted from https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/main/main.py ###
  # This param indicates classifier will be used for
  #   PPLM / NOS-style guidance
  #  (see: https://arxiv.org/abs/2305.20009).

  classifier = classifier.to(model.device)
  # optimizer, _ = classifier.configure_optimizers()
  # optimizer = optimizer[0]
  optimizer = torch.optim.Adam(classifier.classifier_model.output_layer.parameters(), lr=1e-4) # Hardcoded to debug
  criterion = torch.nn.MSELoss()

  if train_config.wandb:
        import wandb
        wandb.init(project="discrete_diffusion", name="classifier_training")
  
  #train manually for now
  n_epochs = train_config.n_epochs
  for epoch in range(n_epochs):
      epoch_loss = 0
      for x, y in dataloader:
          if time_conditioned:
              t = torch.randint(0, model.timestep, (x.shape[0],)).to(torch.long)
          else:
              t = torch.zeros(x.shape[0], dtype=torch.long)
          x = x.to(classifier.device)
          y = y.to(torch.bfloat16).to(classifier.device) #convert to float16
          t = t.to(classifier.device)
          loss = train_step(classifier, model, optimizer, criterion, x, t, y, project_fn, time_conditioned=time_conditioned)
          epoch_loss += loss
          #print(f">Epoch {epoch+1} loss: {loss}\r", end="")
          if train_config.wandb:
              wandb.log({"loss": loss})
      #print(f"Epoch {epoch+1} loss: {epoch_loss/len(dataloader)}")
      if train_config.wandb:
          wandb.log({"epoch_loss": epoch_loss/len(dataloader)})
      
  if train_config.wandb:
      wandb.finish()

  # save model
  if not os.path.exists(save_dir):
      os.makedirs(save_dir)
  torch.save(classifier, os.path.join(save_dir, f"classifier_{ensemble_idx}.pt"))
      
  return classifier


  # trainer = hydra.utils.instantiate(
  #   config.trainer,
  #   default_root_dir=os.getcwd(),
  #   callbacks=callbacks,
  #   strategy=hydra.utils.instantiate(config.strategy),
  #   logger=wandb_logger)
  # trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


