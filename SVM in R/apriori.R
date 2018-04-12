library(arules)
library("arulesViz")
data("Groceries")
inspect(head(Groceries, 3))
frequentItems <- eclat (Groceries, parameter = list(supp = 0.07, maxlen = 15))
itemFrequencyPlot(Groceries, topN=10, type="absolute", main="Item Frequency")
inspect(frequentItems)
rules <- apriori(Groceries, parameter=list(support=0.001, confidence=0.2))
subsetRules <- which(colSums(is.subset(rules, rules)) > 1)
rules <- rules[-subsetRules]
rules
inspect(head(sort(rules, by ="lift"),100))
plot(rules, shading="order", control=list(main = "Two-key plot"))

