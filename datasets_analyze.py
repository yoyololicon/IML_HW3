import forestfires
import iris
import sys

def main():
    print '----for iris dataset----'
    iris.run_iris()
    print '----for forest fires dataset----'
    forestfires.run_forestfires(sys.argv[1])

if __name__ == '__main__':
    main()