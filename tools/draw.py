import numpy as np
import torch
import matplotlib.pyplot as plt

class draw_score():
    def score_field(model, x0, sigma=None, device=None, grid_size=25, xlim=None, ylim=None, normalize=False, ax=None):
        model.eval()

        if isinstance(x0, torch.Tensor):
            data = x0.detach().cpu().numpy()    #取消追蹤梯度，轉cpu，轉numpy
        else:
            data = np.asarray(x0)

        if device is None:
            device = next(model.parameters()).device


        if xlim is None:
            margin = 0.2*np.ptp(data[:,0])
            xlim = (data[:,0].min()-margin, data[:,0].max()+margin)
        if ylim is None:
            margin = 0.5*np.ptp(data[:,1])
            ylim = (data[:,1].min()-margin, data[:,1].max()+margin)

        #造格子
        xx, yy = np.meshgrid(np.linspace(*xlim, grid_size), np.linspace(*ylim, grid_size))

        grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

        grid_tensor = torch.as_tensor(grid, dtype=torch.float32, device=device)

        with torch.no_grad():
            if sigma is None:
                score = model(grid_tensor)
            else:
                sigma_tensor = torch.full(
                    (len(grid_tensor),1),
                    sigma,
                    device=device,
                    dtype=grid_tensor.dtype
                )
                score = model(grid_tensor, sigma_tensor)

            score = score.detach().cpu().numpy()

        if normalize:
            norm = np.linalg.norm(score, axis=1, keepdims=True)
            score = score / np.maximum(norm, 1e-8)


        # fig
        u = score[:, 0].reshape(xx.shape)
        v = score[:, 1].reshape(yy.shape)
        
        plt.figure(figsize=(8,6))

        plt.scatter(
            data[:,0],
            data[:,1],
            s=3,
            alpha=0.25,
            label="Original data"
        )

        plt.quiver(xx, yy, u, v)

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.axis("equal")

        title = "Learned Score Field"
        if sigma is not None:
            title += f" (σ={sigma:g})"

        plt.title(title)
        plt.legend()
        plt.show()



