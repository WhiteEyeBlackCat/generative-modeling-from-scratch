import numpy as np
import math
import torch
import matplotlib.pyplot as plt

class draw_score():
    
    @staticmethod
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

        
        if sigma is None:
            sigmas = [None]
        elif isinstance(sigma, tuple):
            if len(sigma) != 2 or not (0 < sigma[0] < sigma[1]):
                raise ValueError("sigma tuple must be (min_sigma, max_sigma)")
            sigmas = np.geomspace(sigma[0], sigma[1], 6)
        else:
            sigmas = [sigma]

        if ax is not None:
            if len(sigmas) != 1:
                raise ValueError("Don't input single ax when multi_sigma")
            axes = [ax]
            fig = ax.figure
        else:
            fig, axes = plt.subplots(
                1, len(sigmas),
                figsize=(6*len(sigmas), 6),
                squeeze=False
            )
            axes = axes.ravel()


        for s, current_ax in zip(sigmas, axes):
            with torch.no_grad():
                if s is None:
                    score = model(grid_tensor)
                else:
                    sigma_tensor = torch.full(
                        (len(grid_tensor),1),
                        float(s),
                        device=device,
                        dtype=grid_tensor.dtype
                    )
                    score = model(grid_tensor, sigma_tensor)
                score = score.cpu().numpy()

            #把長度固定1
            if normalize:   
                norm = np.linalg.norm(score, axis=1, keepdims=True)
                score = score/np.maximum(norm, 1e-8)


            # fig
            u = score[:, 0].reshape(xx.shape)
            v = score[:, 1].reshape(yy.shape)

            current_ax.scatter(
                data[:,0],
                data[:,1],
                s=3,
                alpha=0.25,
                label="Original data"
            )

            current_ax.quiver(xx, yy, u, v)

            padding = 0.02
            x_pad = (xlim[1] - xlim[0]) * padding
            y_pad = (ylim[1] - ylim[0]) * padding

            current_ax.set_xlim(
                xlim[0] - x_pad,
                xlim[1] + x_pad
            )

            current_ax.set_ylim(
                ylim[0] - y_pad,
                ylim[1] + y_pad
            )
            current_ax.set_aspect("equal")

            title = "Learned Score Field"

            if s is not None:
                title += f" (σ={s:.4g})"

            current_ax.set_title(title)
            current_ax.legend()

        fig.tight_layout()
        return fig, axes 


    def NCSN_infer(history,row=2):
        cols = math.ceil(len(history) / row)
        fig, axes = plt.subplots(row, cols, figsize=(18,7))
        axes = axes.flatten()

        for ax, (sigma, x) in zip(axes, history.items()):
            x = x.cpu().numpy()
            ax.scatter(
                x[:,0],
                x[:,1],
                s=5,
                alpha=0.6
            )

            if sigma == "initial":
                ax.set_title("initial Noise")
            else:
                ax.set_title(f"σ = {sigma:.4f}")
            ax.set_xlim(-3, 4)
            ax.set_ylim(-3, 3)
            ax.set_aspect("equal")

        # 移除多餘的子圖    
        for ax in axes[len(history):]:
            ax.remove()
            
        return fig, axes
