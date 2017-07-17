import sys
from demo import demo
from model import train_model, valid_model

def run_demo(args):
  if len(args) < 1:
    demo('./models')
  else:
    demo(args[1])

def run_model(args):
  if len(args) < 1:
    show_usage()
    exit()
  if args[0] == 'train':
    train_model()
  elif args[0] == 'valid':
    valid_model()
  else:
    show_usage()

def show_usage():
  divide_line = "*-------------------------------------*\n"
  usage = ("|Usage: python3 model.py <train|valid>|\n")
  print(divide_line ,usage, divide_line)


def main():
  args = sys.argv
  if len(args) < 2:
    print("Too few params.")
    print("usage: python3 main.py <function>")
    exit()
  usage = args[1]
  args = args[2:]

  if usage == "demo":
    run_demo(args)
  elif usage == "model":
    run_model(args)
  else :
    print("usage: python3 main.py <function>")

if __name__ == '__main__':
  main()
