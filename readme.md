
Files to replicate the work of my master thesis.

Unfortunately I could not find the time to refactor the whole work in order to make easier to use. But here at least the instructions of how to reproduce the results reported in the mol. inf. article.

to train the models and do the parameter search run:

	analysis/param_search.py config/param_search_base.json

Figure 1 and 4 can be created by:

	analysis/plot/plot_paramsearch.py config/plot_paramsearch_base.json

Figure 2:

	* compute predictions with analysis/compute_predictions.py
	* plot those results with analysis/plot/plot_predictions.py

Figure 3:

	validation/compare_paramselection.py


In case you run into any problems, never hesitate to write me an email:

stephan.gabler.removethisincludingthedot@gmail.com
