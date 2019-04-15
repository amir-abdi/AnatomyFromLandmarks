import torch
from torch import optim
from torch import nn
from common.utils import make_hyparam_string_gen, read_pickles_args, save_voxel_plot
from common.utils import dice, get_data_loaders
import os
import numpy as np
import gen3d


def test(args):
    log_param = make_hyparam_string_gen(args)

    # model define
    model = getattr(gen3d, args.model_name)
    net = model._G(args)

    gpu = args.gpu

    if torch.cuda.is_available():
        print("using cuda")
        net = nn.DataParallel(net, device_ids=[gpu[0]]).to('cuda:' + str(gpu[0]))

    # load model
    if os.path.exists(args.load_model_path):
        read_pickles_args(args, net)
    else:
        raise FileNotFoundError('Trained model not found: {}'.format(args.load_model_path))

    # Prepare datasets
    _, dset_loaders_test = get_data_loaders(data_path=args.data_path_test,
                                            labels_path=args.labels_path_test, args=args)

    dice_scores = np.array([], dtype=np.float)

    image_path = os.path.join(args.output_dir, args.image_dir + '_test', log_param)
    indices = dset_loaders_test.sampler.indices
    num_batches = (len(indices) // args.batch_size)
    print('Saving images in', image_path)

    # =============== validation ===============#
    for j, (shape3d, landmarks) in enumerate(dset_loaders_test):
        generated_shape3d = net(landmarks)

        generated_samples_valid = generated_shape3d.cpu().data.squeeze().numpy()
        real_samples_valid = shape3d.cpu().data.squeeze().numpy()
        landmarks_cpu = landmarks.cpu().data.squeeze().numpy()

        if len(real_samples_valid.shape) == 3:
            real_samples_valid = np.expand_dims(real_samples_valid, axis=0)
            generated_samples_valid = np.expand_dims(generated_samples_valid, axis=0)

        if not os.path.exists(image_path):
            os.makedirs(image_path)

        test_dice_scores_batch = dice(generated_samples_valid, real_samples_valid)
        print('batch:{}/{}, batch dices:{}'.format(j, num_batches, test_dice_scores_batch))
        valid_dice_scores_batch_str = [str(f) for f in test_dice_scores_batch]
        dice_scores = np.concatenate((dice_scores, test_dice_scores_batch))

        if args.save_voxels:
            save_voxel_plot(np.concatenate((real_samples_valid,
                                            generated_samples_valid), axis=0),
                            image_path, args, 'test' + '_' + str(j),
                            titles=valid_dice_scores_batch_str * 2,
                            landmarks=np.concatenate((landmarks_cpu, landmarks_cpu)))

    print(dice_scores)
    test_dice = dice_scores.mean()
    print('test dice:{:.4}'.format(test_dice))

    results_path = os.path.join(image_path, "test_results_{}.csv".format(test_dice))
    num_tested = args.batch_size * num_batches
    np.savetxt(results_path,
               np.stack((indices[:num_tested],
                         dice_scores), axis=-1),
               delimiter=",")
    print('Results saved in', results_path)
