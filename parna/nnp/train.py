import os
from pathlib import Path
package_dir = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.append(str(package_dir))
import argparse
from parna.nnp.utils import train_function, _default_model


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--jitted_model", type=str, required=True)
    
    args = parser.parse_args()
    
    config_yaml_file = args.config
    pretrained_model = args.pretrained_model
    output_model = args.output_model
    jitted_model = args.jitted_model

    train_function(
                config=config_yaml_file, 
                model=_default_model, 
                load=pretrained_model, 
                save=output_model,
                args={}
            )
    os.system(f"aimnet jitcompile {output_model} {jitted_model}")
    print(f"Jitted model saved to {jitted_model}")