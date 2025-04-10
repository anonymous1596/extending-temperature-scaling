import pandas as pd
import plotly.express as px
from IPython.display import display
import numpy as np
def get_df(file_path, model_names, settings, seeds, ood_datasets):
  all = []

  for model_name in model_names:
    for setting in settings:
      for seed in seeds:
        filepath = file_path + '/' + setting + '/' + model_name  + '/' + str(seed) 
        a = pd.read_csv(filepath + '/metrics/calibration.csv', index_col=0)
        a['Model'] = model_name
        a['Setting'] = setting
        a['Seed'] = seed
        for ood_dataset in ood_datasets:
          b = pd.read_csv(filepath + '/metrics/ood_entropy_' + ood_dataset + '.csv', index_col=0)
          b.columns = ood_dataset + '_' + b.columns
          a = pd.concat([a, b], axis = 1)
        all.append(a)
  df = pd.concat(all)
  col_dict = {
    'nll': 'NLL', 'ECE10': 'ECE', 'ECE15': 'ECE (15 Bins)', 'ECE20': 'ECE (20 Bins)'
  }
  df = df.rename(columns=col_dict)
  return df

def get_plot_metric(df, setting, model_names, height = 400, colspace = 0.055, rowspace = 0.04):
  df0 = df[df['Setting'] == setting]
  AUCs_ood = pd.melt(df0, id_vars = ['Model', 'Method', 'Setting'], value_name='Value', var_name='Metric')
  fig = px.box(AUCs_ood, x = 'Method', y = 'Value', 
              #  boxmode="overlay", 
              color = 'Method',
              facet_row='Metric',facet_row_spacing=rowspace, facet_col_spacing=colspace, facet_col='Model',
              category_orders={ 
                'Model': model_names,
                'Method': ['TS', 'SPTS', 'SP1', 'SPU'],
                }
              )

  fig.update_annotations(font_size=20)
  for annotation in fig['layout']['annotations']: 
    annotation['textangle']= 0
    annotation['text'] = ''
  fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True,
                     tickfont=dict(size=16)))
                                                


  r = 0.35
  fig.update_layout(
      autosize=False,
      width=2800*r,
      height=(len(df.columns) - 3)*height*r,
      showlegend=False
  )
  fig.update_xaxes(tickangle=90, tickfont=dict(size=16), title="", showticklabels = False)
  fig.update_yaxes(matches=None, title="")
  return fig
  
def get_csv(df, model_names, Setting):
  df0 = df[df['Setting'] == Setting]
  df1 = df0.groupby(['Model', 'Setting', 'Method']).mean().droplevel('Setting')
  df1 = df1.reset_index()
  df1['Method'] = pd.Categorical(df1['Method'], ['TS', 'SPTS', 'SP1', 'SPU'])
  df1['Model'] = pd.Categorical(df1['Model'], model_names)
  round_n=  3
  df1 = df1.sort_values(['Model', 'Method']).round(round_n)
  df2 = df1.melt(id_vars = ['Model', 'Method'], var_name='Metric', value_name='Value')
  df3 = df2.pivot(columns = 'Model',  index = ['Method', 'Metric'], values = ['Value'])
  df3.columns = df3.columns.droplevel(None)
  df3 = df3.reset_index()
  df3[['OOD', 'Type']] = df3['Metric'].str.split('_', n=1, expand=True)
  df3 = df3.drop('Metric', axis = 1)
  df3.insert(0, 'Type', df3.pop('Type'))
  df3.insert(0, 'OOD', df3.pop('OOD'))
  df3['Type'] = df3['Type'].map({'Difference': 'Avg. Diff', 'Percent Change': 'Avg. % Decr'})
  df3 = df3.rename(columns = {'resnet152': 'Res', 'seresnet152': 'SE', 'xception': 'Xcept', 'densenet161': 'Dense', 'inceptionresnetv2': 'Incept'})
  df4 = pd.concat([df3[df3['Type'] == 'Avg. Diff'].reset_index(), df3[df3['Type'] == 'Avg. % Decr'].reset_index()], axis = 1)
  df4 = df4.iloc[:,[3,1,4,5,6,7,8,13,14,15,16,17]]
  df4 = df4.sort_values('OOD')
  column_tuples = [('', 'Method'), ('', 'OOD')] + [('Avg. Diff', 'Res'), ('Avg. Diff', 'SE'), ('Avg. Diff', 'Xcept'), ('Avg. Diff', 'Dense'), ('Avg. Diff', 'Incept')] + [('Avg. % Decr', 'Res'), ('Avg. % Decr', 'SE'), ('Avg. % Decr', 'Xcept'), ('Avg. % Decr', 'Dense'), ('Avg. % Decr', 'Incept')]
  df4.columns = pd.MultiIndex.from_tuples(column_tuples, names=['Upper', 'Lower'])
  df4[('','OOD')] = df4[('','OOD')].map({'SVHN': 'SVHN', 'places365': 'Places'})
  return df4

  
  
def get_comp(df, method):
  AUCs_ood = pd.melt(df[(df['Setting'].isin(
    [
    'convert', 
    'locshift', 
    'compose'
    ]) &
                        df['Method'].isin([method])
                        ) 
                        ], 
                    id_vars = ['Model', 'Method', 'Setting'], value_name='Value', var_name='Metric')
  AUCs_ood.loc[AUCs_ood['Setting'] == 'compose', 'Setting'] = 'Approach 1: Compose'
  AUCs_ood.loc[AUCs_ood['Setting'] == 'locshift', 'Setting'] = 'Approach 2: LocShift'
  AUCs_ood.loc[AUCs_ood['Setting'] == 'convert', 'Setting'] = 'Approach 3: Convert'

  fig = px.box(AUCs_ood, x = 'Setting', y = 'Value', 
              #  boxmode="overlay", 
              color = 'Setting',
              facet_row='Metric', facet_col_spacing=0.07, facet_col='Model',
              category_orders={ 
                #  'Model': ['ResNet', 'SE-Net', 'Xception', 'DenseNet', 'SqueezeNet'],
                'Setting': ['Approach 1: Compose', 'Approach 2: LocShift', 'Approach 3: Convert']
                }
              )
  fig.update_yaxes(matches=None)
  for annotation in fig['layout']['annotations']: 
    annotation['textangle']= 0
    annotation['text'] = ''
  fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True,
                     tickfont=dict(size=16)))
  r = 0.28
  fig.update_layout(
      autosize=False,
      width=3500*r,
      height=(len(df.columns) - 3)*500*r,
      # yaxis={'title': 'y', 'tickformat':'e', 'rangemode': 'tozero',
      #        'ticks': 'outside'}
  )
  fig.update_xaxes(tickangle=90, showticklabels=False)
  fig.update_layout(title = dict(
    text = '', 
    font = dict(size = 20),
    x = 0.5
  ))
  fig.update_xaxes(tickangle=90, tickfont=dict(size=16), title="", showticklabels = False)
  fig.update_yaxes(matches=None, title="")
  return fig