import os

class Load_dataset:
# works only for IAM dataset
    def load_dataset_cursive(self, source_path):
        pt_path = os.path.join(source_path, "cursive_text_files") 
        paths = {"train": open(os.path.join(pt_path, "trainset.txt")).read().splitlines(),
                "valid": open(os.path.join(pt_path, "validationset.txt")).read().splitlines(),
                "test": open(os.path.join(pt_path, "testset.txt")).read().splitlines()}

        lines = open(os.path.join(source_path, "ascii", "cursive_label.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            split = line.split()

            if split[1] == "ok":
                gt_dict[split[0]] = " ".join(split[8::]).replace("|", " ")

        dataset = dict()

        for i in ['train','test','valid']:
            #dt : stores image
            #gt : stores label for that image
            dataset[i] = {"dt": [], "gt": []}

            for line in paths[i]:
                try:
                    #split = line.split("-")
                    #folder = f"{split[0]}-{split[1]}"

                    img_file = f"{line}.png"
                    img_path = os.path.join(source_path, "cursive", img_file)

                    dataset[i]['gt'].append(gt_dict[line])
                    dataset[i]['dt'].append(img_path)
                except KeyError:
                    pass

        return dataset


    def load_dataset_discrete(self, source_path):
        pt_path = os.path.join(source_path, "discrete_text_files") 
        paths = {"train": open(os.path.join(pt_path, "trainset.txt")).read().splitlines(),
                "valid": open(os.path.join(pt_path, "validationset.txt")).read().splitlines(),
                "test": open(os.path.join(pt_path, "testset.txt")).read().splitlines()}

        lines = open(os.path.join(source_path, "ascii", "discrete_label.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            split = line.split()

            if split[1] == "ok":
                gt_dict[split[0]] = " ".join(split[8::]).replace("|", " ")

        dataset = dict()

        for i in ['train','test','valid']:
            #dt : stores image
            #gt : stores label for that image
            dataset[i] = {"dt": [], "gt": []}

            for line in paths[i]:
                try:
                    #split = line.split("-")
                    #folder = f"{split[0]}-{split[1]}"

                    img_file = f"{line}.png"
                    img_path = os.path.join(source_path, "discrete", img_file)

                    dataset[i]['gt'].append(gt_dict[line])
                    dataset[i]['dt'].append(img_path)
                except KeyError:
                    pass

        return dataset
