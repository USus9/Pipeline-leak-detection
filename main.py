import argparse, os, torch
from core.data_pipeline import DataPipeline
from core.model_trainer import ModelTrainer
from core.leak_detector import LeakDetector
from core.cwt_processor import CWTProcessor
import yaml

NUMERIC_FEATURES = ['VB']  # using VB series only for demo; extend as needed


def train_model(args):
    pipeline = DataPipeline(NUMERIC_FEATURES, sequence_length=args.seq_len)
    df = pipeline.load(args.data_path)
    scal, seq, y = pipeline.prepare_sequences(df)
    trainer = ModelTrainer(args.model_cfg)
    model = trainer.train(scal, seq, y)
    torch.save(model.state_dict(), args.save_path)
    print("Model trained and saved at", args.save_path)


def detect(args):
    # Load model
    with open(args.model_cfg) as f:
        cfg = yaml.safe_load(f)
    seq_len = cfg['training']['sequence_length']
    from core.hybrid_cnn_lstm import HybridCNNLSTM
    arch = cfg['architecture']
    model = HybridCNNLSTM(seq_len, tuple(arch['cnn_filters']), arch['lstm_hidden'], arch['lstm_layers'], arch['dropout'], arch['bidirectional'])
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    cwt = CWTProcessor()
    detector = LeakDetector(model, args.detect_cfg, NUMERIC_FEATURES, cwt)
    pipeline = DataPipeline(NUMERIC_FEATURES, sequence_length=seq_len)
    df = pipeline.load(args.data_path)
    vb_series = df['VB'].values
    leaks = 0
    for vb in vb_series[seq_len:]:
        feat_window = vb_series[len(detector.feature_names):len(detector.feature_names)+seq_len]  # simplistic rolling window
        leak, pred, gap, res = detector.process(feat_window, vb)
        if leak:
            leaks +=1
    print("Total leaks detected", leaks)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')
    t = sub.add_parser('train');
    t.add_argument('--data-path', required=True)
    t.add_argument('--model-cfg', default='config/model_params.yaml')
    t.add_argument('--seq-len', type=int, default=128)
    t.add_argument('--save-path', default='model_best.pth')

    d = sub.add_parser('detect');
    d.add_argument('--data-path', required=True)
    d.add_argument('--model-cfg', default='config/model_params.yaml')
    d.add_argument('--model-path', default='model_best.pth')
    d.add_argument('--detect-cfg', default='config/detection_params.yaml')

    args = p.parse_args()
    if args.cmd == 'train':
        train_model(args)
    elif args.cmd == 'detect':
        detect(args)
    else:
        p.print_help()
