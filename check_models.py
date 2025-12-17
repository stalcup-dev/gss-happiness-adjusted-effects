import statsmodels.api as sm
import statsmodels.genmod as genmod
import statsmodels.discrete as disc

# Check for ordinal models
print("In genmod:")
print([m for m in dir(genmod) if 'order' in m.lower()])

print("\nIn discrete:")
print([m for m in dir(disc) if 'order' in m.lower() or 'logit' in m.lower()])

# Check statsmodels version
print(f"\nstatsmodels version: {sm.__version__}")

# Check if OrderedModel can be imported from different locations
try:
    from statsmodels.genmod.generalized_ordered_model import OrderedModel
    print("OrderedModel available from genmod.generalized_ordered_model")
except ImportError:
    print("OrderedModel NOT available from genmod.generalized_ordered_model")

# Check for alternative ordinal packages
print("\nChecking for ordinal regression alternatives...")
try:
    from mord import LogisticAT
    print("mord package available")
except ImportError:
    print("mord package not available")
