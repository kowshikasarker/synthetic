from pytorch_lightning.callbacks import Callback
from typing_extensions import override
import torch

class MultiLossEarlyStopping(Callback):
    def __init__(self, monitor, min_delta, patience, mode, check_finite, stopping_threshold):
        super().__init__()
        self.mode_dict = {"min": torch.lt, "max": torch.gt}
        self.monitor = monitor
                
        self.min_delta = dict(zip(monitor, min_delta))
        self.patience = dict(zip(monitor, patience))
        self.mode = dict(zip(monitor, mode))
        self.check_finite = dict(zip(monitor, check_finite))
        self.stopping_threshold = dict(zip(monitor, stopping_threshold))
        
        self.best_score = {}
        self.wait_count = {}
        
        torch_inf = torch.tensor(torch.inf)
        
        for m in self.monitor:
            monitor_op = self.mode_dict[self.mode[m]]
            self.min_delta[m] *= 1 if monitor_op == torch.gt else -1
            self.best_score[m] = torch_inf if monitor_op == torch.lt else -torch_inf
            self.wait_count[m] = 0
    
    @override
    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        print('logs', logs, end='\n')
        
        should_stop_count = 0
        for m in self.monitor:
            current = logs[m].squeeze()
            if(self.check_finite[m]):
                if not (torch.isfinite(current)):
                    print("Should stop for", m, "(not finite)")
                    should_stop_count += 1
                    continue
            monitor_op = self.mode_dict[self.mode[m]]
            print('current', current, 'best', self.best_score[m], 'monitor_op', monitor_op)
            
            if(monitor_op(current, self.stopping_threshold[m])):
                print("Should stop for", m, "(stopping_threshold reached)")
                should_stop_count += 1
                continue
            
            if monitor_op(current - self.min_delta[m], self.best_score[m].to(current.device)):
                self.best_score[m] = current
                self.wait_count[m] = 0
            else:
                self.wait_count[m] += 1
                print('self.wait_count', self.wait_count)
                if self.wait_count[m] >= self.patience[m]:
                    print("Should stop for", m, " (not improved)")
                    should_stop_count += 1
        print("should_stop_count", should_stop_count)
        should_stop = should_stop_count == len(self.monitor)
        print('should_stop', should_stop)
        print('trainer.should_stop', trainer.should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        print('trainer.should_stop', trainer.should_stop)
        if should_stop:
            self.stopped_epoch = trainer.current_epoch

class AnnealerStep(Callback):
    def __init__(self):
        super().__init__()
        
    @override
    def on_train_epoch_start(self, trainer, pl_module):
        print()
        print('current_epoch', pl_module.current_epoch, end='\n')
        
    @override
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        print('Param gradients for epoch', pl_module.current_epoch, 'batch', batch_idx, end='\n')
        for name, param in pl_module.model.named_parameters():
            print(name)
            print(param.grad, end='\n')
        print()
        print('Param values for epoch', pl_module.current_epoch, 'batch', batch_idx, end='\n')
        for name, param in pl_module.model.named_parameters():
            print(name)
            print(param, end='\n')
    
    @override
    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.annealer.step()