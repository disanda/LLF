import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import time
from typing import List, Optional
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.generators import Generator
from hydra.utils import instantiate
from pathlib import Path
import utils.training_utils as training_utils
import utils.metrics as metrics
import utils.log_utils as log_utils
from models.generators import stylegan2_ada_networks # StyleGAN2_ada
import pickle
import numpy as np
import joblib
import legacy

class Trainer:
    """Model trainer
    Args:
        model: model to train
        loss_fn: loss function
        optimizer: model optimizer
        generator: pretrained generator
        batch_size: number of batch elements
        iterations: number of iterations
        scheduler: learning rate scheduler
        grad_clip_max_norm: gradient clipping max norm (disabled if None)
        writer: writer which logs metrics to TensorBoard (disabled if None)
        save_path: folder in which to save models (disabled if None)
        checkpoint_path: path to model checkpoint, to resume training
    """
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        iterations: int,
        device: torch.device,
        eval_freq: int = 1000,
        eval_iters: int = 100,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip_max_norm: Optional[float] = None,
        writer: Optional[SummaryWriter] = None,
        save_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        feature_size: Optional[List[int]] = None,
        use_kmeans: bool = False,
        k_th_cluster: int = 5,
        kmeans_model_path:str = "",
        use_mse =  False,
        g_path = None,
        truncation = 0.7
    ) -> None:

        # Logging / Saving  / Device
        self.logger = logging.getLogger()
        self.writer = writer
        self.save_path = save_path
        print(save_path)
        self.device = device

        # Model
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.generator = legacy.load_model(g_path, device)

        #Eval
        self.eval_freq = eval_freq
        self.eval_iters = eval_iters

        # Scheduler
        self.scheduler = scheduler
        self.grad_clip_max_norm = grad_clip_max_norm

        # Batch & Iteration
        self.batch_size = batch_size
        self.iterations = iterations
        self.start_iteration = 0

        # Metrics
        self.train_acc_metric = metrics.LossMetric()
        self.train_loss_metric = metrics.LossMetric()

        self.val_acc_metric = metrics.LossMetric()
        self.val_loss_metric = metrics.LossMetric()

        # Best
        self.best_loss = -1

        self.feature_size = feature_size
        self.truncation = truncation

        # Segmentation
        self.use_kmeans = use_kmeans
        self.use_mse = use_mse
        self.kmeans_model_path = kmeans_model_path
        if use_kmeans == True:
            self.kmeans_model = joblib.load(self.kmeans_model_path) #加载kmeans模型
            # with open(self.kmeans_model_path, "rb") as f:
            #     self.kmeans_model = pickle.load(f)
            self.k_th_cluster = k_th_cluster

        if use_mse == True:
            self.mse_loss = torch.nn.MSELoss()

    def _kmeans_predict(self,feats,kmeans_model):
        """ kmeans预测特征图类别的函数，输出语义分割图。
        Args:
            feats: 从GAN中提取的特征图。
            kmeans_model: 加载的kmeans模型，用于对feats_formask分割。"""
        
        feats_new = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])
        arr = feats_new.detach().cpu().numpy() #检查NAN、无穷大
        arr[np.isnan(arr)]=0
        arr[np.isinf(arr)]=0
        labels = kmeans_model.predict(arr)
        labels_spatial = labels.reshape(feats.shape[0], feats.shape[2], feats.shape[3])
        return labels_spatial

    def train(self) -> None:
        """Trains the model"""
        self.logger.info("Beginning training")
        start_time = time.time()

        epoch = 0
        iteration = self.start_iteration
        while iteration < self.iterations:
            if iteration + self.eval_freq < self.iterations:
                num_iters = self.eval_freq
            else:
                num_iters = self.iterations - iteration

            start_epoch_time = time.time()

            self._train_loop(epoch, num_iters)

            self._val_loop(epoch, self.eval_iters)

            epoch_time = time.time() - start_epoch_time
            self._end_loop(epoch, epoch_time, iteration)

            iteration += num_iters
            epoch += 1

        train_time_h = (time.time() - start_time) / 3600
        self.logger.info(f"Finished training! Total time: {train_time_h:.2f}h")
        self._save_model(
            os.path.join(self.save_path, "model_epoch%d.pt"%epoch), self.iterations
        )

    def _train_loop(self, epoch: int, iterations: int) -> None:
        """ Regular train loop
        Args:
            epoch: current epoch
            iterations: iterations to run model"""
        # Progress bar
        pbar = tqdm.tqdm(total=iterations, leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        # Set to train
        self.model.train()

        # Set to eval
        self.generator.eval()

        for i in range(iterations):

            # Original features
            with torch.no_grad():
                z = self.generator.sample_latent(self.batch_size, self.device, truncation = self.truncation).to(self.device)
                z_orig = z
                orig_feats, _ = self.generator.synthesis(z, mid_size=self.feature_size) # out = out[out_feature_id]

            #获取语义分割
            if self.use_kmeans:
                orig_feats_unchange = orig_feats.clone()
                labels_spatial= self._kmeans_predict(orig_feats_unchange,self.kmeans_model)

            #orig_feats = training_utils.feature_reshape_norm(orig_feats)

            # Apply Directions
            self.optimizer.zero_grad()
            z = self.model(z)

            # Forward
            features = []
            features_unchange = []
            orig_features_unchange = []
            for j in range(z.shape[0] // self.batch_size):
                # Prepare batch
                start, end = j * self.batch_size, (j + 1) * self.batch_size
                z_batch = z[start:end, ...]

                # Get features
                z_batch_label = torch.zeros([end-start, self.generator.c_dim], device=self.device)
                z_batch = self.generator.mapping(z_batch,z_batch_label,truncation_psi=self.truncation)
                feats, _ = self.generator.synthesis(z_batch,mid_size=self.feature_size)

                #备份feats_unchange 预测语义分割图, #获得mask
                if self.use_kmeans:
                    feats_unchange= feats.clone()
                    labels_spatial2 = self._kmeans_predict(feats_unchange,self.kmeans_model)

                #feats = training_utils.feature_reshape_norm(feats)

                # Take feature divergence
                feats = feats - orig_feats

                if self.use_kmeans:
                    mask = (labels_spatial2[:, :, :] == self.k_th_cluster) | (labels_spatial[:, :, :] == self.k_th_cluster)
                    mask = torch.from_numpy(mask).to(self.device)
                    mask = mask.repeat(feats.shape[1],1,1,1).reshape(feats.shape[0],feats.shape[1], feats.shape[2], feats.shape[3]) 
                    feats_unchange.masked_fill_(mask, 0)
                    orig_feats_unchange.masked_fill_(mask, 0)
                    feats.masked_fill_(~mask, 0)
                    features_unchange.append(feats_unchange)             #用于计算MSE的特征，对应编辑后特征图在mask以外的区域
                    orig_features_unchange.append(orig_feats_unchange)   #用于计算MSE的特征，对应原始特征图在mask以外的区域

                #feature_reshape_norm
                feats = torch.reshape(feats, (feats.shape[0], -1))
                feats = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))
                features.append(feats)  #用于对比学习的特征

            features = torch.cat(features, dim=0)
            if self.use_kmeans:
                features_unchange = torch.cat(features_unchange, dim=0)
                orig_features_unchange = torch.cat(orig_features_unchange, dim=0)

            # Loss
            acc, clr_loss = self.loss_fn(features)
            if self.use_mse & self.use_kmeans:
                mse_loss = self.mse_loss(features_unchange,orig_features_unchange)
                loss = clr_loss + 10*mse_loss
            else:
                loss = clr_loss
                mse_loss = torch.tensor(0)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            self.train_acc_metric.update(acc.item(), z.shape[0])
            self.train_loss_metric.update(loss.item(), z.shape[0])

            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(
                f"Acc: {acc.item():.3f} Loss_CLR: {clr_loss.item():.3f} Loss_MSE: {mse_loss.item():.5f}", refresh=False
            )
        pbar.close()

    def _val_loop(self, epoch: int, iterations: int) -> None:
        """Standard validation loop
        Args:
            epoch: current epoch
            iterations: iterations to run model"""
        # Progress bar
        pbar = tqdm.tqdm(total=iterations, leave=False)
        pbar.set_description(f"Epoch {epoch} | Validation")

        # Set to eval
        self.model.eval()
        self.generator.eval()

        # Loop
        for i in range(iterations):
            with torch.no_grad():
                # To device
                z = self.generator.sample_latent(self.batch_size)
                z = z.to(self.device)

                z_label = torch.zeros([self.batch_size, self.generator.c_dim], device=self.device)
                z = self.generator.mapping(z,z_label,truncation_psi=self.truncation)

                # Original features
                orig_feats, _ = self.generator.synthesis(z,self.feature_size)
                orig_feats = training_utils.feature_reshape_norm(orig_feats)
                # Apply Directions
                z = self.model(z)

                # Forward
                features = []
                for j in range(z.shape[0] // self.batch_size):
                    # Prepare batch
                    start, end = j * self.batch_size, (j + 1) * self.batch_size

                    # Get features

                    z_batch_label = torch.zeros([end-start, self.generator.c_dim], device=self.device)
                    z_batch = z[start:end, ...]
                    z_batch = self.generator.mapping(z_batch,z_batch_label,truncation_psi=self.truncation)

                    feats, _ = self.generator.synthesis(z_batch,self.feature_size)
                    feats = training_utils.feature_reshape_norm(feats)
                    # Take feature divergence
                    feats = feats - orig_feats
                    feats = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))

                    features.append(feats)
                features = torch.cat(features, dim=0)

                # Loss
                acc, loss = self.loss_fn(features)
                self.val_acc_metric.update(acc.item(), z.shape[0])
                self.val_loss_metric.update(loss.item(), z.shape[0])

                # Update progress bar
                pbar.update()
                pbar.set_postfix_str(
                    f"Acc: {acc.item():.3f} Loss: {loss.item():.3f}", refresh=False
                )

        pbar.close()

    def _end_loop(self, epoch: int, epoch_time: float, iteration: int):
        # Print epoch results
        self.logger.info(self._epoch_str(epoch, epoch_time))

        # Write to tensorboard
        if self.writer is not None:
            self._write_to_tb(epoch)

        # Save model
        eval_loss = self.val_loss_metric.compute()
        if self.best_loss == -1 or eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self._save_model(os.path.join(self.save_path, "most_recent_epoch%d_best.pt"%epoch), iteration)
        elif self.save_path is not None:
            if epoch% 5 == 0:
                self._save_model(os.path.join(self.save_path, "most_recent_epoch%d.pt"%epoch), iteration)
        else:
            print('do not save model')
            
        # Clear metrics
        self.train_loss_metric.reset()
        self.train_acc_metric.reset()
        self.val_loss_metric.reset()
        self.val_acc_metric.reset()

    def _epoch_str(self, epoch: int, epoch_time: float):
        s = f"Epoch {epoch} "
        s += f"| Train acc: {self.train_acc_metric.compute():.3f} "
        s += f"| Train loss: {self.train_loss_metric.compute():.3f} "
        s += f"| Val acc: {self.val_acc_metric.compute():.3f} "
        s += f"| Val loss: {self.val_loss_metric.compute():.3f} "
        s += f"| Epoch time: {epoch_time:.1f}s"
        return s

    def _write_to_tb(self, iteration):
        self.writer.add_scalar(
            "Loss/train", self.train_loss_metric.compute(), iteration
        )
        self.writer.add_scalar("Acc/train", self.train_acc_metric.compute(), iteration)
        self.writer.add_scalar("Loss/val", self.val_loss_metric.compute(), iteration)
        self.writer.add_scalar("Acc/val", self.val_acc_metric.compute(), iteration)

    def _save_model(self, path, iteration):
        obj = {
            "iteration": iteration + 1,
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,}
        torch.save(obj, os.path.join(self.save_path, path))

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_iteration = checkpoint["iteration"]

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if self.start_iteration > self.iterations:
            raise ValueError("Starting iteration is larger than total iterations")

        self.logger.info(f"Checkpoint loaded, resuming from iteration {self.start_iteration}")

@hydra.main(config_path="./utils/training_config", config_name="train_direction_model_stylegan2_ada")
def train(cfg: DictConfig):
    # Init model
    model_save_path = os.getcwd()
    log_utils.display_config(cfg)
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k).to(device)
    loss_fn: torch.nn.Module = instantiate(cfg.loss, k=cfg.k).to(device)
    optimizer: torch.optim.Optimizer = instantiate(
        cfg.optimizer,
        model.parameters(),
    )
    scheduler = instantiate(cfg.scheduler, optimizer)

    # Tensorboard
    if cfg.tensorboard:
        # Note: global step is in epochs here
        writer = SummaryWriter(os.getcwd())
        # Indicate to TensorBoard that the text is pre-formatted
        text = f"<pre>{OmegaConf.to_yaml(cfg)}</pre>"
        writer.add_text("config", text)
    else:
        writer = None

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        batch_size=cfg.batch_size,
        iterations=cfg.iterations,
        device=device,
        eval_freq=cfg.eval_freq,
        eval_iters=cfg.eval_iters,
        scheduler=scheduler,
        grad_clip_max_norm=cfg.grad_clip_max_norm,
        writer=writer,
        save_path=model_save_path,
        checkpoint_path=cfg.model_load_path,
        feature_size=cfg.generator.feature_size,
        use_kmeans = cfg.kmeans.use_kmeans,
        k_th_cluster = cfg.kmeans.k_th_cluster,
        kmeans_model_path= hydra.utils.to_absolute_path(cfg.kmeans.kmeans_model_path),
        use_mse =  cfg.kmeans.use_mse_loss,
        g_path = hydra.utils.to_absolute_path(cfg.generator.generator_path),
        truncation = cfg.generator.truncation

    )     # Trainer init

    trainer.train() # Launch training process

if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print('#######################')
    train()


