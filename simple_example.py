from code.FigureSeparator import FigureSeparator
fig_separator=FigureSeparator("./data/figure-sepration-model-submitted-544.pb")
sub_figures=fig_separator.extract("./imgs/PMC4076561-Figure5-1.png")
print(sub_figures)