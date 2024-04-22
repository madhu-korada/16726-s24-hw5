# Original paper: https://arxiv.org/abs/2112.10752 
# Original Authors: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer
# Assignment Coordinator: Hariharan Ravichandran


import argparse, os
import torch
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from util import load_img, load_model_from_config
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar", help="the prompt to render")
    parser.add_argument("--input-img", type=str, nargs="?", default="assets/sketch-cat-512.png", help="path to the input image")
    parser.add_argument("--num_timesteps", type=int, default=500, help="number of ddpm sampling steps")
    parser.add_argument("--strength", type=float, default=15.0, help="guidance strength")
    parser.add_argument("--seed", type=int, default=10, help="the seed (for reproducible sampling)")

    return parser.parse_args()

def main():
    
    # Parse arguments
    opt = parse_args()
    assert opt.strength > 1.0
    
    # Set seed
    seed_everything(opt.seed)
    
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the prompt
    prompt = opt.prompt
    assert prompt is not None

    # Load the input image
    assert os.path.isfile(opt.input_img)
    input_image = load_img(opt.input_img).to(device)

    # Load the model
    config = OmegaConf.load(f"configs/inference.yaml")
    model = load_model_from_config(config, f"models/model.ckpt").to(device)

    # Define the timesteps
    timesteps = np.asarray(list(range(0, opt.num_timesteps)))
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    timesteps += 1

    # Alphas for DDPM
    alphas = (1. - model.betas)
    alpha_cumprods = model.alphas_cumprod
    
    #############
    ####TODO#####
    #############
    with torch.no_grad():
        # Move input image to latent space
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(input_image))
        # Set the unconditional conditioning
        uncond = model.get_learned_conditioning([""])
        
        # Use the prompt to get the learned conditioning
        cond = model.get_learned_conditioning([prompt])
        
        # TODO: Generate noise tensor for the input image
        noise = torch.randn_like(init_latent)
        
        # TODO: Add noise to the latent space and get the encoded latent state
        latent = torch.sqrt(alpha_cumprods[-1]) * init_latent + (torch.sqrt(1 - alpha_cumprods[-1]) * noise)


        # TODO: Reverse the timesteps for denoising
        reversed_time_range = reversed(timesteps)        

        # TODO: Initialize the latent state for DDPM sampling
        # latent = init_latent
        
        
        out_dir = "outputs/" + prompt.replace(" ", "_") + "/"
        os.makedirs(out_dir, exist_ok=True)
        
        # Loop over the reversed time steps
        for i, timestep in tqdm(enumerate(reversed_time_range)):            
            # Timestep tensor for the current step 
            # timestep = torch.full(size=(1,), fill_value=timestep,  device=device, dtype=torch.long)
            timestep_tensor = torch.full(size=(1,), fill_value=timestep,  device=device, dtype=torch.long)
            # TODO: Get the score estimator for the conditional and unconditional guidance
            # Hint 1: Use the apply_model function which returns the score estimator for the conditional and unconditional guidance
            #      The function takes in three arguments: x_in, t_in, c_in
            #      x_in: The latent state
            #      t_in: The timestep tensor
            #      c_in: The guidance tensor
            # Hint 2: The guidance tensor is the concatenation of the unconditional and conditional guidance.
            #         So you need to repeat the latent and timestep tensors for the two guidance scores as well
            # Hint 3: Use the chunk function to separate the score estimator after applying the model
            #         to separate the conditional and unconditional guidance
            x_in = repeat(latent, 'b ... -> (repeat b) ...', repeat=2)
            t_in = repeat(timestep_tensor, 'b -> (repeat b)', repeat=2)
            c_in = torch.cat([uncond, cond], dim=0)
            
            e_t_combined = model.apply_model(x_in, t_in, c_in)
            e_t_uncond, e_t_cond = torch.chunk(e_t_combined, 2, dim=0)
            
            # TODO: Calculate the classifier-free diffusion guidance score
            e_t = e_t_uncond + opt.strength * (e_t_cond - e_t_uncond)
            
            # TODO: Update the latent state using DDPM Sampling
            sigma_t = torch.sqrt(((1 - alpha_cumprods[timestep - 1]) / (1 - alpha_cumprods[timestep])) * (1 - alphas[timestep]))
            # noise = torch.randn_like(init_latent)
            noise = torch.randn_like(latent) if i < (opt.num_timesteps - 1) else torch.zeros_like(latent)

            latent = (1 / torch.sqrt(alphas[timestep - 1])) * (latent - ((1 - alphas[timestep]) / torch.sqrt(1 - alpha_cumprods[timestep])) * e_t) + sigma_t * noise

            # Save the output image at 10% of the total timesteps
            if i % (opt.num_timesteps // 10) == 0:
                # Get the decoded sample from the first stage
                output_image = model.decode_first_stage(latent)
                output_image = torch.clamp((output_image + 1.0) / 2.0, min=0.0, max=1.0)
                out_image = 255. * rearrange(output_image[0].cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(out_image.astype(np.uint8)).save(out_dir + f"{i:05}.png")
            
        # Get the decoded sample from the first stage
        output_images = model.decode_first_stage(latent)
        print("output_images.shape: ", output_images.shape)
        
        # Clamp the output images from [-1, 1] to [0, 1]
        output_images = torch.clamp((output_images + 1.0) / 2.0, min=0.0, max=1.0)
        print("output_images.shape after clamp: ", output_images.shape)
        
        image_name = prompt.replace(" ", "_") + ".png"
        # Save images
        for i, out_image in enumerate(output_images):
            out_image = 255. * rearrange(out_image.cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(out_image.astype(np.uint8)).save("outputs/" + image_name)

    print("Done!")

if __name__ == "__main__":
    main()