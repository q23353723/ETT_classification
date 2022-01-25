import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

to_correct = [
	[3589,	'1.2.826.0.1.3680043.8.498.57005638787237813934531972491254580369',	'CVC - Borderline',	'NGT - Borderline'],
	[4344,	'1.2.826.0.1.3680043.8.498.93345761486297843389996628528592497280',	'ETT - Abnormal',	'CVC - Abnormal'],
	[6294,	'1.2.826.0.1.3680043.8.498.50891603479257167332052859560303996365',	'NGT - Normal',	'CVC - Normal'],
	[7558,	'1.2.826.0.1.3680043.8.498.32665013930528750130301395098139968929',	'NGT - Borderline',	'CVC - Borderline'],
	[8457,	'1.2.826.0.1.3680043.8.498.47822809495672253227315400926882161159',	'NGT - Borderline',	'CVC - Borderline'],
	[8586,	'1.2.826.0.1.3680043.8.498.55171965195784371324650309161724846475',	'NGT - Borderline',	'CVC - Borderline'],
	[8589,	'1.2.826.0.1.3680043.8.498.29639870594803047496855371142714987539',	'ETT - Normal',	'CVC - Normal'],
	[9908,	'1.2.826.0.1.3680043.8.498.52422864792637441690285442425747003963',	'NGT - Normal',	'ETT - Normal'],
	[10889,	'1.2.826.0.1.3680043.8.498.51277351337858188519077141427236143108',	'NGT - Normal',	'CVC - Normal'],
	[10963,	'1.2.826.0.1.3680043.8.498.33011244702337270174558484639492100815',	'CVC - Normal',	'NGT - Normal'],
	[11902,	'1.2.826.0.1.3680043.8.498.10505287747515183956922280117689383476',	'NGT - Normal',	'CVC - Normal'],
	[12041,	'1.2.826.0.1.3680043.8.498.43340424479611237895060478106689360500',	'NGT - Normal',	'CVC - Normal'],
	[12782,	'1.2.826.0.1.3680043.8.498.12545979153892772426852721449004507757',	'NGT - Abnormal',	'CVC - Abnormal'],
	[13513,	'1.2.826.0.1.3680043.8.498.83700037297895094021306651705503600111',	'NGT - Normal',	'ETT - Normal'],
	[14226,	'1.2.826.0.1.3680043.8.498.35772244095675958072394978496245125294',	'NGT - Normal',	'ETT - Normal'],
	[15750,	'1.2.826.0.1.3680043.8.498.96130195933728659348647733812659169362',	'CVC - Abnormal',	'NGT - Abnormal'],
	[15779,	'1.2.826.0.1.3680043.8.498.75269816256944932004789976844599885553',	'NGT - Abnormal',	'CVC - Abnormal'],
	[16629,	'1.2.826.0.1.3680043.8.498.11935284122896798228836385959451625327',	'NGT - Abnormal',	'CVC - Abnormal'],
	[17501,	'1.2.826.0.1.3680043.8.498.83574817573978660270935463700320068005',	'NGT - Abnormal',	'CVC - Abnormal']
]

def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break
    
    weight = module.weight.detach()
    module.in_channels = new_in_channels
    
    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(
                module.out_channels,
                new_in_channels // module.groups,
                *module.kernel_size
            )
        )
        module.reset_parameters()
    
    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)
    
    else:
        new_weight = torch.Tensor(
            module.out_channels,
            new_in_channels // module.groups,
            *module.kernel_size
        )

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)

def do_correct(df):
    for case in to_correct:
        df.loc[df.StudyInstanceUID==case[1], case[2]] = 0
        df.loc[df.StudyInstanceUID==case[1], case[3]] = 1
    
    return df

def load_csv(dir):
    df = pd.read_csv(dir)
    df = do_correct(df)
    df = df[['StudyInstanceUID', 'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal']]
    df = df[(df['ETT - Abnormal'] == 1) | (df['ETT - Borderline'] == 1) | (df['ETT - Normal'] == 1)]

    return df

def get_weights(dataset):
    df_class_0 = dataset[dataset['ETT - Abnormal'] == 1]['ETT - Abnormal']
    df_class_1 = dataset[dataset['ETT - Borderline'] == 1]['ETT - Borderline']
    df_class_2 = dataset[dataset['ETT - Normal'] == 1]['ETT - Normal']

    df_class_1 = df_class_1.add(1)
    df_class_2 = df_class_2.add(2)
    df_total = pd.concat([df_class_0, df_class_1, df_class_2], axis=0)

    class_weights = compute_class_weight('balanced', np.unique(df_total), df_total.values)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return class_weights

def upsampling(dataset):
    df_class_0 = dataset[dataset['ETT - Abnormal'] == 1]
    df_class_1 = dataset[dataset['ETT - Borderline'] == 1]
    df_class_2 = dataset[dataset['ETT - Normal'] == 1]

    num = max([len(df_class_0), len(df_class_1), len(df_class_2)])
    df_class_0_over = resample(df_class_0, replace=True, n_samples=num, random_state=42)
    df_class_1_over = resample(df_class_1, replace=True, n_samples=num, random_state=42) 
    train = pd.concat([df_class_0_over, df_class_1_over, df_class_2], axis=0)

    return train

def undersampling(dataset):
    df_class_0 = dataset[dataset['ETT - Abnormal'] == 1]
    df_class_1 = dataset[dataset['ETT - Borderline'] == 1]
    df_class_2 = dataset[dataset['ETT - Normal'] == 1]

    num = max([len(df_class_0), len(df_class_1), len(df_class_2)])
    df_class_1_under = resample(df_class_1, replace=True, n_samples=num, random_state=42)
    df_class_2_under = resample(df_class_2, replace=True, n_samples=num, random_state=42) 
    train = pd.concat([df_class_0, df_class_1_under, df_class_2_under], axis=0)

    return train

def hybridsampling(dataset, sample_num):
    df_class_0 = dataset[dataset['ETT - Abnormal'] == 1]
    df_class_1 = dataset[dataset['ETT - Borderline'] == 1]
    df_class_2 = dataset[dataset['ETT - Normal'] == 1]

    df_class_0 = resample(df_class_0, replace=True, n_samples=sample_num, random_state=42)
    df_class_1 = resample(df_class_1, replace=True, n_samples=sample_num, random_state=42)
    df_class_2 = resample(df_class_2, replace=True, n_samples=sample_num, random_state=42)
    train = pd.concat([df_class_0, df_class_1, df_class_2], axis=0)

    return train