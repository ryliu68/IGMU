from utils.multiC_evaluator import Evaluator
import argparse
import glob
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    evaluator = Evaluator(indicator=args.indicator, args=args, device=device)

    DATA_PATH = "dataset/Benchmarking_images_demo/Nudity/forgot/ORG"
    filenames = glob.glob(F"{DATA_PATH}/*[.png|.jpg|.jpeg|.JPEG]")

    acc = evaluator.eval(filenames)

    print(args.indicator, args.concept, acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for SafeGuide benchmark")
    parser.add_argument("--concept", type=str, default="Nudity", required=False, help="Support concept eval belong to 'Nudity', 'Style (129 classes, e.g., 'vincent-van-gogh' )' and 'Object (10 classes, e.g., 'church')', detailed concept name and id refer to utils/name_to_id.py")
    parser.add_argument("--indicator", type=str, default="multi_multiC", required=False, choices=["bi_multiC", "multi_multiC"])
    parser.add_argument('--batch-size', type=int, default=5)
    args = parser.parse_args()

    main(args)