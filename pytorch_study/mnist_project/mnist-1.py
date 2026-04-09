import argparse

def train():
    print("开始训练...")

def test():
    print("开始测试...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    print("你设置的 epochs =", args.epochs)

    for epoch in range(1, args.epochs + 1):
        print(f"\n第 {epoch} 轮")
        train()
        test()

if __name__ == "__main__":
    main()