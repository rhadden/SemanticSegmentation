import os


def build_set(input_root, output_file_name):
    samples = []
    cities = os.listdir(input_root)
    for city in cities:
        city_dir = input_root + os.sep + city
        files = [city_dir + os.sep + x for x in os.listdir(city_dir) if x.endswith("leftImg8bit.png")]
        samples += files

    with open(output_file_name, 'w+') as f:
        for sample in samples:
            parts = sample.split(os.sep)
            parts[-4] = 'gtFine'
            parts[-1] = parts[-1][:-15] + 'gtFine_labelIds.png'
            labels = os.sep.join(parts)
            f.write(sample + ',' + labels + "\n")


def main():
    build_set(root + "/train", "train.csv")
    build_set(root + "/test", "test.csv")
    build_set(root + "/val", "val.csv")

if __name__=="__main__":
    # root = '/home/riley/anaconda3/gtFine_trainvaltest/gtFine'
    root = '/home/riley/anaconda3/gtFine_trainvaltest/leftImg8bit'
    main()
