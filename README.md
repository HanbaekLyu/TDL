# Temporal Dictionary Learning 
Dictionary Learning from joint time serieses using Online Matrix Factorization (flexible nonnegativity constraints on dictioanry and code matrices)
Learns dictionary atoms for short time-evolution patterns of multiple entities and uses them to reconstruct the time-series data

## References
These codes are based on the following papers
  1. Hanbaek Lyu, Christopher Strohmeier, Deanna Needell, and Georg Menz, 
    “COVID-19 Time Series Prediction by Joint Dictionary Learning and Online NMF” 
    https://arxiv.org/abs/2004.09112

  2. Hanbaek Lyu, Palina Salanevich, Jacob Li, Charlotte Huang, and  Deanna Needell
    "Temporal Dictionary Learning for EEG and Constructing Correlation Tensor"
    In preperation.

## File description 
  1. **utils/TDL.py** : Main file implementing temporal dictionary learning
  2. **utils/TDL_plotting.py** : Helper functions for plotting
  3. **utils/onmf.py** : Online Nonnegative Matrix Factorization algorithm (generalization of onmf to the tensor setting by folding/unfolding operation)
  4. **covid_dataprocess.py** : Preprocessing functions (modify this for your own data type)
  5. **TDL-COVID-Test.ipynb** : Jupyter notebook example of temporal dictionary learning

## Author

* **Hanbaek Lyu** - *Initial work* - [Website](https://hanbaeklyu.com)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
