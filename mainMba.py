# %%
import pandas as pd
import os
from mlxtend.preprocessing import TransactionEncoder 
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth






# %%
df_ventas = pd.read_csv("ventas.csv")

# %%
doc_producto_dict = df_ventas.groupby('tDocumento')['tDetallado'].apply(list).to_dict()
# Obtener listas de listas de los productos por factura
dataset = list(doc_producto_dict.values())
dataset


# %%
# Obtengo los productos para realizar combinaciones a partir de los productos principales
cortes = df_ventas[df_ventas['Subgrupo']=='PARRILLA']
cortes = cortes['tDetallado'].tolist()
# Función para verificar si un conjunto de ítems contiene al menos un producto de cortes
def contiene_cortes(itemset):
    return any(item in cortes for item in itemset)

# %%
te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_array, columns=te.columns_)
df

# %%
frequent_itemsets_fp=fpgrowth(df, min_support=0.02, use_colnames=True)

# %%
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.2)

# %%
rules_fp

# %%
productos_carnicos = frozenset(cortes)
# Filtrar las reglas donde los antecedentes contienen al menos un producto de cortes
filtered_rules = rules_fp[rules_fp['antecedents'].apply(contiene_cortes)]
filtered_rules

# %%
frequent_itemsets_fp


