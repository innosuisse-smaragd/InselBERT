import evaluate

# TODO: Add perplexity, average word length

mcnemar = evaluate.load("mcnemar")
results = mcnemar.compute(references=[1, 0, 1], predictions1=[1, 1, 1], predictions2=[1, 0, 1])
print(results)