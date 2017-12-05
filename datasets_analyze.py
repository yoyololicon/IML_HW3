import forestfires
import iris

def main():
    print 'for iris dataset'
    iris.run_iris()
    print 'for forest fires dataset'
    forestfires.run_forestfires()

if __name__ == '__main__':
    main()