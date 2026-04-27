from torch.utils.data import DataLoader
from src.sdk.dataset import SignalDataset
from src.models.rnn import RNNFilter
from src.sdk.gatekeeper import APIGatekeeper
from src.sdk.trainer import SignalTrainer

def test_gatekeeper_config_loading():
    model = RNNFilter()
    gk = APIGatekeeper(model)
    assert "max_requests_per_second" in gk.config
    assert gk.config["version"] == "1.0.0"

def test_trainer_epoch():
    dataset = SignalDataset(num_samples=1, window_size=10)
    dataloader = DataLoader(dataset, batch_size=2)
    
    model = RNNFilter(hidden_dim=8)
    gk = APIGatekeeper(model)
    trainer = SignalTrainer(gk, lr=0.01)
    
    initial_loss = trainer.train_epoch(dataloader)
    assert isinstance(initial_loss, float)
    assert initial_loss >= 0
