import torch
import torch.nn as nn
import torch.nn.functional as F
from .valuation_func import *


class YOLOValuationModule(nn.Module):
    """A module to call valuation functions.
        Attrs:
            lang (language): The language.
            device (device): The device.
            layers (list(nn.Module)): The list of valuation functions.
            vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
            attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
            dataset (str): The dataset.
    """

    def __init__(self, lang, device, dataset):
        super().__init__()
        self.lang = lang
        self.device = device
        self.layers, self.vfs = self.init_valuation_functions(device, dataset)
        # attr_term -> vector representation dic
        self.attrs = self.init_attr_encodings(device)
        self.dataset = dataset

    def init_valuation_functions(self, device, dataset):
        """
            Args:
                device (device): The device.
                dataset (str): The dataset.

            Retunrs:
                layers (list(nn.Module)): The list of valuation functions.
                vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
        """
        layers = []
        vfs = {}  # a dictionary: pred_name -> valuation function
        v_color = YOLOColorValuationFunction()
        vfs['color'] = v_color
        layers.append(v_color)

        v_sort = YOLOsortValuationFunction()
        vfs['sort'] = v_sort
        layers.append(v_sort)

        v_move = YOLOmoveValuationFunction()
        vfs['move'] = v_move
        layers.append(v_move)

        v_step_ver = YOLOstepVerValuationFunction()
        vfs['step_ver'] = v_step_ver
        layers.append(v_step_ver)

        v_step_hori = YOLOstepHoriValuationFunction()
        vfs['step_hori'] = v_step_hori
        layers.append(v_step_hori)

        v_shape = YOLOShapeValuationFunction()
        vfs['shape'] = v_shape
        v_in = YOLOInValuationFunction()
        vfs['in'] = v_in
        layers.append(v_in)
        v_closeby = YOLOClosebyValuationFunction(device)
        if dataset in ['closeby', 'red-triangle']:
            vfs['closeby'] = v_closeby
            vfs['closeby'].load_state_dict(torch.load(
                'src/weights/neural_predicates/closeby_pretrain.pt', map_location=device))
            vfs['closeby'].eval()
            layers.append(v_closeby)
            print('Pretrained  neural predicate closeby have been loaded!')
        elif dataset == 'online-pair':
            v_online = YOLOOnlineValuationFunction(device)
            vfs['online'] = v_online
            vfs['online'].load_state_dict(torch.load(
                'src/weights/neural_predicates/online_pretrain.pt', map_location=device))
            vfs['online'].eval()
            layers.append(v_online)
            print('Pretrained  neural predicate online have been loaded!')
        return nn.ModuleList(layers), vfs

    def init_attr_encodings(self, device):
        """Encode color and shape into one-hot encoding.

            Args:
                device (device): The device.

            Returns:
                attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
        """
        attr_names = ['color', 'shape', 'direction', 'steps_ver', 'steps_hori','group']
        attrs = {}
        for dtype_name in attr_names:
            for term in self.lang.get_by_dtype_name(dtype_name):
                term_index = self.lang.term_index(term)
                num_classes = len(self.lang.get_by_dtype_name(dtype_name))
                one_hot = F.one_hot(torch.tensor(
                    term_index).to(device), num_classes=num_classes)
                one_hot.to(device)
                attrs[term] = one_hot
        return attrs

    def forward(self, zs, atom):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representaion (the output of the YOLO model).
                atom (atom): The target atom to compute its proability.

            Returns:
                A batch of the probabilities of the target atom.
        """
        if atom.pred.name in self.vfs:
            args = [self.ground_to_tensor(term, zs) for term in atom.terms]
            # call valuation function
            return self.vfs[atom.pred.name](*args)
        else:
            return torch.zeros((zs.size(0), )).to(
                torch.float32).to(self.device)

    def ground_to_tensor(self, term, zs):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        term_index = self.lang.term_index(term)
        if term.dtype.name == 'object':
            return zs[:, term_index]
        elif term.dtype.name == 'color' or term.dtype.name == 'shape':
            return self.attrs[term]
        elif term.dtype.name == 'direction':
            return self.attrs[term]
        elif term.dtype.name == 'steps_ver':
            return self.attrs[term]
        elif term.dtype.name == 'steps_hori':
            return self.attrs[term]
        elif term.dtype.name == 'group':
            return self.attrs[term]
        elif term.dtype.name == 'image':
            return None
        else:
            assert 0, "Invalid datatype of the given term: " + \
                str(term) + ':' + term.dtype.name


class SlotAttentionValuationModule(nn.Module):
    """A module to call valuation functions.
        Attrs:
            lang (language): The language.
            device (device): The device.
            layers (list(nn.Module)): The list of valuation functions.
            vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
            attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
            dataset (str): The dataset.
    """

    def __init__(self, lang, device, pretrained=True):
        super().__init__()
        self.lang = lang
        self.device = device
        self.colors = ["cyan", "blue", "yellow",
                       "purple", "red", "green", "gray", "brown"]
        self.shapes = ["sphere", "cube", "cylinder"]
        self.sizes = ["large", "small"]
        self.materials = ["rubber", "metal"]
        self.sides = ["left", "right"]

        self.layers, self.vfs = self.init_valuation_functions(
            device, pretrained)

    def init_valuation_functions(self, device, pretrained):
        """
            Args:
                device (device): The device.
                pretrained (bool): The flag if the neural predicates are pretrained or not.

            Retunrs:
                layers (list(nn.Module)): The list of valuation functions.
                vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
        """
        layers = []
        vfs = {}  # pred name -> valuation function
        v_color = SlotAttentionColorValuationFunction(device)
        vfs['color'] = v_color
        v_shape = SlotAttentionShapeValuationFunction(device)
        vfs['shape'] = v_shape
        v_in = SlotAttentionInValuationFunction(device)
        vfs['in'] = v_in
        v_size = SlotAttentionSizeValuationFunction(device)
        vfs['size'] = v_size
        v_material = SlotAttentionMaterialValuationFunction(device)
        vfs['material'] = v_material
        v_rightside = SlotAttentionRightSideValuationFunction(device)
        vfs['rightside'] = v_rightside
        v_leftside = SlotAttentionLeftSideValuationFunction(device)
        vfs['leftside'] = v_leftside
        v_front = SlotAttentionFrontValuationFunction(device)
        vfs['front'] = v_front

        if pretrained:
            vfs['rightside'].load_state_dict(torch.load(
                'src/weights/neural_predicates/rightside_pretrain.pt', map_location=device))
            vfs['rightside'].eval()
            vfs['leftside'].load_state_dict(torch.load(
                'src/weights/neural_predicates/leftside_pretrain.pt', map_location=device))
            vfs['leftside'].eval()
            vfs['front'].load_state_dict(torch.load(
                'src/weights/neural_predicates/front_pretrain.pt', map_location=device))
            vfs['front'].eval()
            print('Pretrained  neural predicates have been loaded!')
        return nn.ModuleList([v_color, v_shape, v_in, v_size, v_material, v_rightside, v_leftside, v_front]), vfs

    def forward(self, zs, atom):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representaion (the output of the YOLO model).
                atom (atom): The target atom to compute its proability.

            Returns:
                A batch of the probabilities of the target atom.
        """
        # term: logical term
        # arg: vector representation of the term
        # zs = self.preprocess(zs)
        args = [self.ground_to_tensor(term, zs) for term in atom.terms]
        # call valuation function
        return self.vfs[atom.pred.name](*args)

    def ground_to_tensor(self, term, zs):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.
        """
        term_index = self.lang.term_index(term)
        if term.dtype.name == 'object':
            return zs[:, term_index]
        elif term.dtype.name == 'image':
            return None
        else:
            # other attributes
            return self.term_to_onehot(term, batch_size=zs.size(0))

    def term_to_onehot(self, term, batch_size):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        if term.dtype.name == 'color':
            return self.to_onehot_batch(self.colors.index(term.name), len(self.colors), batch_size)
        elif term.dtype.name == 'shape':
            return self.to_onehot_batch(self.shapes.index(term.name), len(self.shapes), batch_size)
        elif term.dtype.name == 'material':
            return self.to_onehot_batch(self.materials.index(term.name), len(self.materials), batch_size)
        elif term.dtype.name == 'size':
            return self.to_onehot_batch(self.sizes.index(term.name), len(self.sizes), batch_size)
        elif term.dtype.name == 'side':
            return self.to_onehot_batch(self.sides.index(term.name), len(self.sides), batch_size)
        else:
            assert True, 'Invalid term: ' + str(term)

    def to_onehot_batch(self, i, length, batch_size):
        """Compute the one-hot encoding that is expanded to the batch size.
        """
        onehot = torch.zeros(batch_size, length, ).to(self.device)
        onehot[:, i] = 1.0
        return onehot

class SquareClassifier(nn.Module):
    def __init__(self):
        super(SquareClassifier, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 16 output channels, 3x3 kernel
       # self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Second convolutional layer: 16 input channels, 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
       # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

      #  # Fully connected layer: After two 2x2 MaxPools, the image is reduced to 32 * 8 * 8
      #  self.fc1 = nn.Linear(32 * 8 * 8, 64)  # Hidden layer with 64 neurons
      #  self.fc2 = nn.Linear(64, 1)  # Final output layer for binary classification

        self.fc1 = nn.Linear(64 * 8 * 8, 64)  # Hidden layer with 32 neurons
        self.fc2 = nn.Linear(64, 1)

        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First convolutional layer followed by ReLU and 2x2 MaxPool
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Output shape: (batch_size, 16, 16, 16)

        # Second convolutional layer followed by ReLU and 2x2 MaxPool
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Output shape: (batch_size, 32, 8, 8)

       # x = F.relu(self.conv3(x))
       # x = F.max_pool2d(x, 2)

        # Flatten the feature maps into a vector
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 32 * 8 * 8 = 2048)

        # Pass through the first fully connected layer + ReLU + Dropout
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)

        # Final output layer + Sigmoid for binary classification
        x = torch.sigmoid(self.fc2(x))

        return x