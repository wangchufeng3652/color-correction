import torch
from torch.utils.data import DataLoader
from models.unet_film_lab import LiteUNetFiLMResidualLab
from utils.dataset import ImageVectorDatasetLab


device=torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


dataset=ImageVectorDatasetLab(
    "data/example/input",
    "data/example/target",
    "data/example/vectors.json",
    device
)


loader=DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)


model=LiteUNetFiLMResidualLab().to(device)


optimizer=torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


loss_fn=torch.nn.MSELoss()


for epoch in range(100):

    model.train()

    loss_all=0


    for img,vec,target in loader:


        pred=model(
            img,
            vec
        )


        loss=loss_fn(
            pred,
            target
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        loss_all+=loss.item()



    print(
    epoch,
    loss_all/len(loader)
    )


torch.save(
    model.state_dict(),
    "weights/best_model.pth"
)
