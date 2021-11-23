import os
import torch
import collections
import shutil

def average_checkpoints(inputs, output):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    print('averaging checkpoints in {} ...'.format(inputs))

    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for fpath in inputs:
        state = torch.load(
            fpath,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(fpath, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state = averaged_params

    torch.save(new_state, output)
    print('avg checkpoint has saved at {}'.format(output))

if __name__ == '__main__':

    model_names = ['roberta']#, 'bert', 'electra']
    seeds = [111, 222, 333, 444, 555, 666, 777, 888]#, 999, 1111]

    for model_name in model_names:
        to_avg_path_list = []
        output_path = './checkpoints/{}'.format(model_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for seed in seeds:
            to_avg_path_list.append('./checkpoints/{}/{}/pytorch_model.bin'.format(model_name, seed))
        average_checkpoints(to_avg_path_list, output=output_path+'/pytorch_model.bin')

        # 从第一个seed文件夹里复制一下config.json和vocab.txt
        shutil.copyfile(src='./checkpoints/{}/{}/config.json'.format(model_name, seeds[0]),
                        dst='./checkpoints/{}/config.json'.format(model_name))
        shutil.copyfile(src='./checkpoints/{}/{}/vocab.txt'.format(model_name, seeds[0]),
                        dst='./checkpoints/{}/vocab.txt'.format(model_name))