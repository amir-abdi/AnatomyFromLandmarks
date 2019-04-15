import os
import numpy as np
import wandb

import torch
from torch import optim
from torch import nn
from torch.optim import lr_scheduler

from common.utils import make_hyparam_string_gen, save_new_pickle, save_voxel_plot
from common.utils import var_or_cuda, get_data_loaders
from common.utils import dice, loss_function, read_pickles_args
from common.torch_utils import dice_torch
import gen3d


def train(args):
    log_param = make_hyparam_string_gen(args)

    # for using tensorboard
    if args.use_tensorboard:
        import tensorflow as tf
        summary_writer = tf.summary.FileWriter(os.path.join(args.output_dir,
                                                            args.tb_log_dir,
                                                            log_param))

        def inject_summary(summary_writer, tag, value, step):
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            summary_writer.add_summary(summary, global_step=step)

        inject_summary = inject_summary

    # Prepare datasets
    dset_loaders_train, dset_loaders_valid = get_data_loaders(args.data_path, args.labels_path, args)

    # GPU device to use
    gpu = args.gpu

    # model define
    model = getattr(gen3d, args.model_name)
    net = model._G(args)

    if torch.cuda.is_available():
        print("using cuda")
        net = nn.DataParallel(net, device_ids=[gpu[0]]).to('cuda:' + str(gpu[0]))

    solver = optim.Adam(net.parameters(), lr=args.g_lr, betas=args.beta, weight_decay=args.weight_decay)

    if args.lrsh is not None:
        # todo(amirabdi): different args are needed for different schedulers
        G_scheduler = getattr(lr_scheduler, args.lrsh)(solver, gamma=args.lr_gamma_g)  # 0.7

    criterion_reconst = loss_function(args.gen_g_loss[0], wbce_weights=args.wbce_weights, focal_gamma=args.focal_gamma)

    # load saved models
    read_pickles_args(args, net, solver)

    if args.use_wandb:
        wandb.watch(net)

    try:
        iteration = solver.state_dict()['state'][solver.state_dict()['param_groups'][0]['params'][0]]['step']
        epoch = int(iteration / (dset_loaders_train.__len__() / args.batch_size))
        print('Continuing from epoch', epoch)
    except:
        epoch = 0
        print('Start training from scratch')

    print("[------ Training ------]")
    while epoch < args.num_epochs:
        net.train()
        for i, (shape3d, landmarks) in enumerate(dset_loaders_train):
            shape3d = var_or_cuda(shape3d, gpu[0])
            landmarks = var_or_cuda(landmarks, gpu[0])

            if shape3d.size()[0] != int(args.batch_size):
                continue

            # =============== Train the generator ===============#
            generated_shape3d = net(landmarks)
            X_expand = shape3d.view(-1, 1, args.cube_len, args.cube_len, args.cube_len)

            g_reconst_loss = criterion_reconst(generated_shape3d, X_expand)
            g_loss = g_reconst_loss

            # if current batch_size doesn't fit in memory, accumulate over multiple samples
            g_loss.backward()
            solver.step()
            net.zero_grad()

            dice_scores_torch = dice_torch(generated_shape3d, X_expand)
            iteration = solver.state_dict()['state'][solver.state_dict()['param_groups'][0]['params'][0]]['step']

            # =============== save generated train images ===============#
            num_view_samples = min(args.num_view_samples_per_batch, args.batch_size)
            if iteration % args.image_save_step == 0:
                image_path = os.path.join(args.output_dir, args.image_dir + "_train", log_param)
                print('Saving generated train images in {} with iteration {}'.format(image_path, iteration))

                generated_samples = generated_shape3d.cpu().data[:num_view_samples].squeeze().numpy()
                real_samples = shape3d.cpu().data[:num_view_samples].squeeze().numpy()
                landmarks_cpu = landmarks.cpu().data[:num_view_samples].squeeze().numpy()

                if len(real_samples.shape) == 3:
                    real_samples = np.expand_dims(real_samples, axis=0)
                    generated_samples = np.expand_dims(generated_samples, axis=0)

                dice_scores_str = [str(d) for d in dice(generated_samples, real_samples)]
                all_samples = np.concatenate((real_samples, generated_samples), axis=0)

                if not os.path.exists(image_path):
                    os.makedirs(image_path)

                save_voxel_plot(all_samples, image_path, args, str(iteration),
                                titles=dice_scores_str * 2,
                                landmarks=np.concatenate((landmarks_cpu, landmarks_cpu)))

            # =============== logging iterations ===============#
            if iteration % args.log_step == 0:
                info = {
                    'loss/loss_G_reconst': g_reconst_loss.data.item(),
                    'loss/loss_G': g_reconst_loss.data.item(),
                    'dice': dice_scores_torch.data.item()
                }
                if args.use_tensorboard:
                    log_save_path = os.path.join(args.output_dir, args.tb_log_dir, log_param)
                    if not os.path.exists(log_save_path):
                        os.makedirs(log_save_path)
                    for tag, value in info.items():
                        inject_summary(summary_writer, tag, value, str(iteration))
                    summary_writer.flush()

                if args.use_wandb:
                    wandb.log(info)

                print(
                    'Epoch:{}, Iter-{} , G_loss: {:.4}, g_reconst_loss: {:.4}, '
                    'G_lr:{:.4}, dice:{:.3}'.format(
                        epoch, iteration,
                        g_loss.data.item(), g_reconst_loss.data.item(),
                        solver.state_dict()['param_groups'][0]["lr"],
                        dice_scores_torch.data.item()))

        # =============== save model as pickle ===============#
        if (epoch + 1) % args.pickle_epoch == 0:
            pickle_save_path = os.path.join(args.output_dir, args.pickle_dir, log_param)
            save_new_pickle(pickle_save_path, str(iteration), net, solver, iter_append=args.save_by_iter)
            print('Model saved in', pickle_save_path)

        # =============== validation ===============#
        if (epoch + 1) % args.valid_epoch == 0:
            print("[------ Validation ------]")
            net.eval()
            image_path = os.path.join(args.output_dir, args.image_dir + "_valid", log_param)
            valid_dice_scores = np.array([], dtype=np.float)
            with torch.no_grad():
                for j, (shape3d, landmarks) in enumerate(dset_loaders_valid):
                    shape3d = var_or_cuda(shape3d, gpu[0])
                    generated_shape3d = net(landmarks)

                    generated_samples_valid = generated_shape3d.cpu().data.squeeze().numpy()
                    real_samples_valid = shape3d.cpu().data.squeeze().numpy()
                    landmarks_cpu = landmarks.cpu().data.squeeze().numpy()

                    if len(real_samples_valid.shape) == 3:
                        real_samples_valid = np.expand_dims(real_samples_valid, axis=0)
                        generated_samples_valid = np.expand_dims(generated_samples_valid, axis=0)

                    if not os.path.exists(image_path):
                        os.makedirs(image_path)

                    valid_dice_scores_batch = dice(generated_samples_valid, real_samples_valid)
                    valid_dice_scores_batch_str = [str(f) for f in valid_dice_scores_batch]
                    valid_dice_scores = np.concatenate((valid_dice_scores, valid_dice_scores_batch))

                    save_voxel_plot(np.concatenate((real_samples_valid[:num_view_samples],
                                                    generated_samples_valid[:num_view_samples]), axis=0),
                                    image_path, args, str(iteration) + '_' + str(j),
                                    titles=valid_dice_scores_batch_str[:num_view_samples] * 2,
                                    landmarks=np.concatenate((landmarks_cpu[:num_view_samples],
                                                              landmarks_cpu[:num_view_samples])),
                                    mode='valid')
            print('Validation images saved in', image_path)
            print(valid_dice_scores)
            valid_dice = valid_dice_scores.mean()
            valid_info = {"valid_dice": valid_dice}
            print('valid dice:{:.4}'.format(valid_dice))
            if args.use_wandb:
                wandb.log(valid_info)
            if args.use_tensorboard:
                for tag, value in valid_info.items():
                    inject_summary(summary_writer, tag, value, str(iteration))
                summary_writer.flush()
            print("[-----------------------]")

        # =============== set learning rate ===============#
        if args.lrsh:
            try:
                G_scheduler.step(epoch)

            except Exception as e:
                print("fail lr scheduling", e)

        if args.use_wandb:
            wandb.log()
        epoch += 1
    print("[---- Training complete ----]")
