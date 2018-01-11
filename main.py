import argparse

from demo import demo
from model import train_model, valid_model


def run_demo(args):
  demo(args.modelPath, args.showbox)

def run_model(args):
  if args.train_or_valid == 'train':
    train_model()
  elif args.train_or_valid == 'valid':
    valid_model()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("func", type=str,
                      help="Choose a func you want to run. <Demo> or <model>")
  parser.add_argument("--modelPath", default="./models", type=str,
                      help="Specify the path to models.")
  parser.add_argument("--showbox", action="store_true",
                      help="Options decide the box of faces whether to show.")
  parser.add_argument("train_or_valid", default="train", type=str,
                      help="Train or va")
  args = parser.parse_args()
  func = args.func
  print(args)
  if func == "demo":
    run_demo(args)
  elif func == "model":
    run_model(args)
  else :
    print("usage: python3 main.py <function>")

if __name__ == '__main__':
  main()
