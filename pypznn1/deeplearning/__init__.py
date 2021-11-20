is_simple_core = False

if is_simple_core:
    from pypznn1.deeplearning.core_simple import Variable
    from pypznn1.deeplearning.core_simple import Function
    from pypznn1.deeplearning.core_simple import using_config
    from pypznn1.deeplearning.core_simple import no_grad
    from pypznn1.deeplearning.core_simple import as_array
    from pypznn1.deeplearning.core_simple import as_variable
    from pypznn1.deeplearning.core_simple import setup_variable
else:
    from pypznn1.deeplearning.core import Variable
    from pypznn1.deeplearning.core import Parameter
    from pypznn1.deeplearning.core import Function
    from pypznn1.deeplearning.core import using_config
    from pypznn1.deeplearning.core import no_grad
    from pypznn1.deeplearning.core import test_mode
    from pypznn1.deeplearning.core import as_array
    from pypznn1.deeplearning.core import as_variable
    from pypznn1.deeplearning.core import setup_variable
    from pypznn1.deeplearning.core import Config
    from pypznn1.deeplearning.layers import Layer
    from pypznn1.deeplearning.models import Model
    from pypznn1.deeplearning.datasets import Dataset
    from pypznn1.deeplearning.dataloaders import DataLoader
    from pypznn1.deeplearning.dataloaders import SeqDataLoader

    import pypznn1.deeplearning.datasets
    import pypznn1.deeplearning.dataloaders
    import pypznn1.deeplearning.functions
    import pypznn1.deeplearning.functions_conv
    import pypznn1.deeplearning.layers
    import pypznn1.deeplearning.optimizers
    import pypznn1.deeplearning.datasets
    import pypznn1.deeplearning.utils
    import pypznn1.deeplearning.transforms
    import pypznn1.deeplearning.cuda

setup_variable()
