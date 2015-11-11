import sys
import matplotlib.pyplot as plt
import seaborn

def main():
    """docstring for main"""
    file = sys.argv[1]
    f = open(file, 'r+')
    last_line = ""
    train_err = []
    for line in f.readlines():
        if line.strip() == "Saving... Done":
            cost = last_line.strip().split()[-1]
            train_err.append(float(cost))
        last_line = line
    f.close()

    with open(file) as f:
        lines = f.readlines()
    valid_err = [float(x) for x in lines[-6].strip()[1:-1].split(',')]
    test_err = [float(x) for x in lines[-4].strip()[1:-1].split(',')]

    plt.figure(figsize=(10, 8))
    plt.plot(train_err, label="Train Error")
    plt.plot(valid_err, label="Valid Error")
    plt.plot(test_err, label="Test Error")
    plt.ylabel("Error")
    plt.legend(frameon=True, prop={"size": 15})
    plt.savefig(sys.argv[2] + ".png")

if __name__ == '__main__':
    main()
