import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description="Model Params")
    parser.add_argument("--model", type=str, default="ALDI")
    parser.add_argument(
        "--reg",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Normal batch size."
    )
    parser.add_argument("--data", default="warm", type=str, help="data")
    parser.add_argument("--freq_coef_a", default=10, type=float, help="coef_a")
    parser.add_argument("--freq_coef_M", default=4, type=float, help="coef_M")
    parser.add_argument("--dataset", default="Amazon2018", type=str, help="dataset")
    parser.add_argument(
        "model", choices=["ALDI", "Recommender"], help="Specify the model to run."
    )
    parser.add_argument(
        "action",
        help="Specify the action to perform (e.g., 'train_teacher', 'test_teacher', etc.).",
    )
    return parser.parse_args()


args = ParseArgs()
