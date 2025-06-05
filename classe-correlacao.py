import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt 

class CorrelacaoSpearman:
    def __init__(self, df, y):
        self.df = df.copy()
        self.y = y
        self.resultado = None

    def calcular(self):
        y_series = self.df[self.y]
        resultados = []

        for coluna in self.df.select_dtypes(include='number').columns:
            if coluna != self.y:
                rho, pval = spearmanr(self.df[coluna], y_series, nan_policy='omit')
                resultados.append({
                    'variavel': coluna,
                    'correlacao_spearman': rho,
                    'p_valor': pval
                })

        self.resultado = pd.DataFrame(resultados)
        self.resultado['abs_corr'] = self.resultado['correlacao_spearman'].abs()
        self.resultado = self.resultado.sort_values(by='abs_corr', ascending=False).drop(columns='abs_corr')


    def plot_heatmap(self, top_n=10):
      if self.resultado is None:
        print("Você precisa rodar o método calcular() antes.")

      top_vars = self.resultado.head(top_n)['variavel'].tolist()
      top_vars.append(self.y)

      corr = self.df[top_vars].corr(method='spearman')

      plt.figure(figsize=(10, 8))
      sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
      plt.title(f'Heatmap de Correlação de Spearman (Top {top_n} Variáveis)')
      plt.tight_layout()
      plt.show()

    def exibir(self, top_n=10):
        if self.resultado is None:
            print("Você precisa rodar o método calcular() antes.")
        else:
            print(self.resultado.head(top_n))