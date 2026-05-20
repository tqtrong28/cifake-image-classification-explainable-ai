import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self.save_activations
        )
        self.backward_hook = self.target_layer.register_full_backward_hook(
            self.save_gradients
        )

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, target_class="real"):
        self.model.eval()

        self.model.zero_grad()

        output = self.model(image_tensor)

        if target_class == "real":
            score = output[:, 0]
        elif target_class == "fake":
            score = -output[:, 0]
        else:
            raise ValueError("target_class must be 'real' or 'fake'")

        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.detach().cpu()

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
