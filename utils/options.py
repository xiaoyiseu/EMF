import argparse

def get_args():
    parser = argparse.ArgumentParser(description="AI Triage")
    ######################## trandformer settings ########################
    parser.add_argument("--embed_dim", default=128)
    parser.add_argument("--num_heads", default=4)
    parser.add_argument("--hidden_dim", default=256)
    parser.add_argument("--nec", default=3, help = 'num_encoder_layers')
    parser.add_argument("--ndc", default=3, help = 'num_decoder_layers')

    ######################## imputation settings ########################
    parser.add_argument("--ImputMode", default='', 
                        help="'RF', 'GAN', 'MICE', 'VAE'")    
 
    ######################## general settings ########################
    parser.add_argument("--backbone", default='TextResNet', 
                        help="'Transformer', 'ResNet', 'TextCNN', 'CrossAtt', 'TextResNet'")
    parser.add_argument("--grade", default=True, help="Graded training")
    parser.add_argument("--SA", default=True, help="Self-Ateention")
    parser.add_argument("--SFD", default=True, help="Manhattan Distance Feature")  


    parser.add_argument("--filt", default=False, help="FFM")# FFM  #False, True
    parser.add_argument("--CA", default=True, help="Cross-Ateention")
    parser.add_argument("--CMF", default=True, help="Cross-Modal Fusion")
    parser.add_argument("--loss", default='', # pdc+ctl+cmc
                        help="which loss to use ['pdc', 'ctl', 'cmc']") 
    
    ######################## file path settings ########################
    parser.add_argument("--data_path", default=r"./data/TriageData.txt")
    parser.add_argument("--stopword_path", default=r"./StopWords/stopword.txt")
    parser.add_argument("--cache_dir", default=r'./cached_data/')
    parser.add_argument("--bestmodel", default=r'weight/best_model.pth')
    parser.add_argument("--vsmodel", default=r'weight/vs_model.pth')
    parser.add_argument("--log_path", default=r'./result/train/training_log.txt')
    parser.add_argument("--output_dir", default=r'./result/')
    parser.add_argument("--indices_path", default=r'./data/data_indices.txt')

    ######################## model general settings ####################
    parser.add_argument("--length", type=int, default=5000000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--bs_test", type=int, default=512)
    parser.add_argument("--quantile", type=float, default=0.3)

    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.2)   
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--mode", default='train')
    args = parser.parse_args(args = [])
    return args
