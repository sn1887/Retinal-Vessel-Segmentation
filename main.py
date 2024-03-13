import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--n-channels', type=int, default=1, help='Number of input channels')

    return parser.parse_args()


#-------------------------------------------------------------------------------------------------------------------------------------

args = get_args()
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    #model = Iternet(n_channels=1, n_classes=1, out_channels=32, iterations=3)
    model = VisionTransformer(TRANSCONFIG['R50-ViT-B_16'], img_size=256, num_classes=1, zero_head=False, vis=True)
    #model = Attention(TRANSCONFIG['ViT-B_16'], vis = False)
    #model = UNet_base(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    #model = Metapolyp(in_chans = 1)
    model = model.to(memory_format=torch.channels_last)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")
    # Use DataParallel to wrap the model

    logging.info(f'Network:\n'
                 f'\t{args.n_channels} input channels\n'
                 f'\t{args.classes} output channels (classes)\n'
                 f'\t{"Bilinear" if args.bilinear else "Transposed conv"} upscaling')
    model = nn.DataParallel(model)
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device)
    try:
        train_model(
            model=model,
            args = args,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )