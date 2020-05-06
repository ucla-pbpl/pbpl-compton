import train_image_energy_reusable
import predict_summed_image_reusable
import eval_model_reusable
import plot_12_plots_reusable
import argparse

parser = argparse.ArgumentParser(
    description='Train and evaluate network based on data file')
parser.add_argument("--config", required=True, 
    help="grand config file")
args = parser.parse_args()
train_image_energy_reusable.main(args.config)
eval_model_reusable.main(args.config)
plot_12_plots_reusable.main(args.config)