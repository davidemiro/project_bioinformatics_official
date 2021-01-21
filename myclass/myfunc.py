

scaler = StandardScaler()
minmaxScaler = MinMaxScaler()

# normalization of the dataset to calculate the variance
X_min_max= minmaxScaler.fit_transform(X)
X_scaler = scaler.fit_transform(X_min_max)

# calculate the pca and the realtive variances
pca = PCA()
df_X_pca = pca.fit_transform(X_scaler)

tot = sum(pca.explained_variance_)
# amount of variance for each components
var_exp = [(i/tot) * 100 for i in sorted(pca.explained_variance_, reverse=True)] 
cum_var_exp = np.cumsum(var_exp) # cumulative sum

trace_cum_var_exp = go.Bar(
	x=list(range(1, len(cum_var_exp) + 1)), 
	y=var_exp,
	name="Individual variance",
)

trace_ind_var_exp = go.Scatter(
	x=list(range(1, len(cum_var_exp) + 1)),
	y=cum_var_exp,
	mode='lines+markers',
	name="Cumulative variance",
	line=dict(
		shape='hv',
	))
		
data = [trace_cum_var_exp, trace_ind_var_exp]
layout = go.Layout(
		title='Variance and PCA',
		autosize=True,
		yaxis=dict(
			title='Percentage of variance',
		),
		xaxis=dict(
			title="n_components PCA",
			dtick=1,
		),
		legend=dict(
			x=0,
			y=1,
		),
	)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='basic-bar')

