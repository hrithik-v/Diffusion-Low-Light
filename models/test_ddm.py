import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import os
import unittest
import torch
from torchinfo import summary
from models.ddm import Net, DenoisingDiffusion
import yaml
from argparse import Namespace

class TestDDM(unittest.TestCase):
    def setUp(self):
        args = None  # Replace with actual arguments if needed
        with open("configs/lsui.yml", 'r') as file:  # Replace with the actual path to your config file
            self.config = yaml.safe_load(file)
        self.model = Net(args, self.config)
        self.ddm = DenoisingDiffusion(args, self.config)


    def test_number_of_parameters(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_params}")
        self.assertTrue(num_params > 0)

    def test_forward_pass(self):
        input_tensor = torch.randn(1, 3, 256, 256)
        output = self.model(input_tensor)
        self.assertIsInstance(output, dict)
        self.assertIn("pred_x", output)
        self.assertEqual(output["pred_x"].shape, (1, 3, 256, 256))

    def test_model_summary(self):
        summary(self.model, input_size=(1, 3, 256, 256))
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


        def dict2namespace(config):
            namespace = Namespace()
            for key, value in config.items():
                if isinstance(value, dict):
                    setattr(namespace, key, dict2namespace(value))
                else:
                    setattr(namespace, key, value)
            return namespace

        class TestDDM(unittest.TestCase):
            def setUp(self):
                args = Namespace(
                    config='configs/lsui.yml',
                    resume='',
                    sampling_timesteps=10,
                    image_folder='results/',
                    seed=230
                )
                with open("configs/lsui.yml", 'r') as file:  # Replace with the actual path to your config file
                    config_dict = yaml.safe_load(file)
                self.config = dict2namespace(config_dict)
                self.model = Net(args, self.config)
                self.ddm = DenoisingDiffusion(args, self.config)

            def test_number_of_parameters(self):
                num_params = sum(p.numel() for p in self.model.parameters())
                print(f"Number of parameters: {num_params}")
                self.assertTrue(num_params > 0)

            def test_forward_pass(self):
                input_tensor = torch.randn(1, 3, 256, 256)
                output = self.model(input_tensor)
                self.assertIsInstance(output, dict)
                self.assertIn("pred_x", output)
                self.assertEqual(output["pred_x"].shape, (1, 3, 256, 256))

            def test_model_summary(self):
                summary(self.model, input_size=(1, 3, 256, 256))

            def test_net_initialization(self):
                self.assertIsInstance(self.model, Net)

            def test_net_forward_pass(self):
                input_tensor = torch.randn(1, 3, 256, 256)
                output = self.model(input_tensor)
                self.assertIsInstance(output, dict)
                self.assertIn("pred_x", output)
                self.assertEqual(output["pred_x"].shape, (1, 3, 256, 256))

            def test_sample_training(self):
                def dict2namespace(config):
                    namespace = Namespace()
                    for key, value in config.items():
                        if isinstance(value, dict):
                            setattr(namespace, key, dict2namespace(value))
                        else:
                            setattr(namespace, key, value)
                    return namespace

                class TestDDM(unittest.TestCase):
                    def setUp(self):
                        args = Namespace(
                            config='configs/lsui.yml',
                            resume='',
                            sampling_timesteps=10,
                            image_folder='results/',
                            seed=230
                        )
                        with open("configs/lsui.yml", 'r') as file:  # Replace with the actual path to your config file
                            config_dict = yaml.safe_load(file)
                        self.config = dict2namespace(config_dict)
                        self.model = Net(args, self.config)
                        self.ddm = DenoisingDiffusion(args, self.config)

                    def test_number_of_parameters(self):
                        num_params = sum(p.numel() for p in self.model.parameters())
                        print(f"Number of parameters: {num_params}")
                        self.assertTrue(num_params > 0)

                    def test_forward_pass(self):
                        input_tensor = torch.randn(1, 3, 256, 256)
                        output = self.model(input_tensor)
                        self.assertIsInstance(output, dict)
                        self.assertIn("pred_x", output)
                        self.assertEqual(output["pred_x"].shape, (1, 3, 256, 256))

                    def test_model_summary(self):
                        summary(self.model, input_size=(1, 3, 256, 256))

                    def test_net_initialization(self):
                        self.assertIsInstance(self.model, Net)

                    def test_net_forward_pass(self):
                        input_tensor = torch.randn(1, 3, 256, 256)
                        output = self.model(input_tensor)
                        self.assertIsInstance(output, dict)
                        self.assertIn("pred_x", output)
                        self.assertEqual(output["pred_x"].shape, (1, 3, 256, 256))

                    def test_sample_training(self):
                        input_tensor = torch.randn(1, 3, 256, 256)
                        b = torch.randn(self.model.num_timesteps)
                        output = self.model.sample_training(input_tensor, b)
                        self.assertEqual(output.shape, (1, 3, 256, 256))

            def test_ddm_initialization(self):
                self.assertIsInstance(self.ddm, DenoisingDiffusion)

            def test_ddm_train_method(self):
                # This is a placeholder test to ensure the train method can be called
                # You should replace DATASET with an actual dataset object
                class DummyDataset:
                    def get_loaders(self):
                        return [], []

                dataset = DummyDataset()
                try:
                    self.ddm.train(dataset)
                except Exception as e:
                    self.fail(f"train method raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()