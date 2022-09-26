# biosim
Supervised biological similarity models built for distribution with biobricks


# Development
```sh
python3 -m venv venv
source ./venv/bin/activate 
pip install -r requirements.txt
```

# TODO
1. Create better latent space models (VAE, GCN)
2. Condition latent space on assay
3. Illustrate that you can move 'towards' or 'away' from an assay value
4. Write paper

# Directory Structure

`src` - code used to build resources  
`src/models`  
`src/generators` 

`notebook` - some scratch code for analyzing models  

`datadep` - imported data dependences