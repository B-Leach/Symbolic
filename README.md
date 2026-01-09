# Symbolic Learning

An interactive web application that uses **genetic programming** to discover mathematical formulas through symbolic regression. Watch in real-time as the algorithm evolves a population of equations to match your target function.

![Symbolic Learning Demo](https://img.shields.io/badge/Live%20Demo-symbolic--learning.benleach.com-blue)

## 🌟 Features

### Custom Equation Input
Enter any mathematical expression and watch the genetic algorithm learn it:
- **Functions**: `sin`, `cos`, `tan`, `arctan`, `sqrt`, `log`, `exp`, `abs`
- **Operators**: `+`, `-`, `*`, `/`, `**` (or `^`)
- **Constants**: `pi`, `e`
- **Variable**: `x`

### Real-Time Visualization
Watch the evolution process with animated plots showing:
- Target function
- Best predicted function per generation
- Live prediction progress
- R² score evolution over time

### Advanced Genetic Programming
Powered by [gplearn](https://gplearn.readthedocs.io/) with optimized parameters:
- Population size: 300 individuals
- 30 generations of evolution
- Tournament selection with size 20
- Carefully tuned mutation and crossover rates
- Protected mathematical operators for numerical stability

### Safe Equation Parser
Equations are parsed using Python's AST (Abstract Syntax Tree) for security:
- No arbitrary code execution
- Validated mathematical expressions only
- Client-side validation before training

## 🚀 Live Demo

Visit [symbolic-learning.benleach.com](http://symbolic-learning.benleach.com) to try it out!

## 📸 Screenshots

### Example Results

**Simple Trigonometric Function**
```
Target: sin(x)
Result: sin(x)
R² Score: 1.0 | MAE: 0.0
```

**Complex Expression**
```
Target: 2*sin(x) + cos(2*x)
Result: sin(x) / cos(6.639) / cos(cos(6.639)) + sin(x) + cos(6.639) + ...
R² Score: 0.992 | MAE: 0.1048
```

**Polynomial**
```
Target: x^2 + 3*x - 5
Result: x * x - -2.070 - 5.728 ** 0.940 - x
R² Score: 0.9998 | MAE: 0.3679
```

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Genetic Programming**: gplearn
- **Visualization**: Matplotlib with h264 encoding
- **Frontend**: Vanilla JavaScript with modern CSS
- **Deployment**: uWSGI + Nginx

## 📦 Installation

### Prerequisites
- Python 3.10+
- FFmpeg (for video encoding)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd symbolic-learning.benleach.com
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install flask gplearn matplotlib numpy
```

4. Ensure FFmpeg is installed:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

5. Run the application:
```bash
python main.py
```

6. Open your browser to `http://localhost:5000`

## 🔧 Configuration

Key parameters can be adjusted in `main.py`:

```python
# Population and evolution
pop_size = 300              # Population size
gen_amt = 30                # Number of generations
tournament_size = 20        # Tournament selection size

# Genetic operators
p_crossover = 0.7           # Crossover probability
p_subtree_mutation = 0.1    # Subtree mutation probability
p_hoist_mutation = 0.05     # Hoist mutation probability
p_point_mutation = 0.1      # Point mutation probability

# Training
max_samples = 0.9           # Fraction of training data to use
stopping_criteria = 0.01    # Stop if fitness reaches this threshold
```

## 🧮 How It Works

1. **User Input**: Enter a mathematical equation (e.g., `2*sin(x) + cos(2*x)`)

2. **Validation**: The equation is parsed and validated using Python's AST parser

3. **Function Detection**: The system automatically detects which mathematical functions are used

4. **Evolution**: 
   - Initialize a population of random equations
   - Evaluate fitness (Mean Absolute Error)
   - Select best performers via tournament selection
   - Create new generation through crossover and mutation
   - Repeat for 30 generations

5. **Visualization**: Real-time animated plots show the evolution process

6. **Results**: The best equation is displayed with R² score and MAE

## 📊 Genetic Programming Details

### Protected Functions
To ensure numerical stability, we use protected versions of potentially problematic operations:

- **Protected Division**: Returns 1 when denominator is near zero
- **Protected Power**: Clamps exponents to (-5, 5) range
- **Protected Square Root**: Uses `sqrt(abs(x))`
- **Protected Logarithm**: Uses `log(abs(x) + 1e-10)`

### Fitness Metric
Uses Mean Absolute Error (MAE) without wrapping for accurate gradient information.

### Parsimony
Low parsimony coefficient (0.001) allows complex solutions while still preferring simpler ones when fitness is equal.

## 🎨 UI Features

- **Dark Theme**: Modern dark UI matching current web design trends
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Feedback**: Validation errors shown immediately
- **Interactive Controls**: Randomize button for example equations
- **Replay Animation**: Rewatch the evolution process

## 🔒 Security

- Safe equation parsing using AST (no `eval()` or `exec()`)
- Whitelist of allowed functions and operators
- Input validation on both client and server side
- Protected mathematical operations prevent numerical errors

## 📝 API Endpoints

### `GET /`
Returns the main application page

### `POST /validate`
Validates an equation before training
```json
{
  "equation": "sin(x) + cos(x)"
}
```

### `POST /train`
Trains the model on an equation
```json
{
  "equation": "2*sin(x) + cos(2*x)"
}
```

Returns:
```json
{
  "success": true,
  "r2_score": 0.992,
  "mae_score": 0.1048,
  "pr_eq_formatted": "sin(x) + ...",
  "video_html": "<video>...</video>",
  "target_equation": "2*sin(x) + cos(2*x)"
}
```

## 🤝 Contributing

Contributions are welcome! Some ideas:
- Add more mathematical functions (sinh, cosh, etc.)
- Implement multi-variable regression
- Add equation simplification
- Improve animation rendering
- Add export functionality for discovered equations

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [gplearn](https://gplearn.readthedocs.io/) - Genetic Programming library
- [Matplotlib](https://matplotlib.org/) - Plotting library
- [Flask](https://flask.palletsprojects.com/) - Web framework

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Built with ❤️ using Genetic Programming**
