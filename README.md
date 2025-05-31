# HERMES: Healthcare Ethics & Robustness in Medical Image Systems

## Description

HERMES is a project focused on ensuring healthcare ethics and robustness in medical image systems.

## Badges

On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals

Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation

> [!TIP]
> We strongly encourage using [astral-sh/uv](https://github.com/astral-sh/uv) to install the project requirements. This ensures that the correct versions of the dependencies are installed, and it can help avoid conflicts with other projects. Alternatively, you can use `pip` to install the requirements from the `requirements.txt` file.

```bash
uv sync
```

## Usage

The project uses Lightning on top of PyTorch, as to reduce boilerplate code and make it easier to write complex models. The project is structured to allow for easy experimentation with different settings, training strategies, and data preprocessing techniques.

To run the project, you can use the following command:

```bash
python run.py
```

it will automatically load the configuration file `default.yaml` in the config directory and start training the model. If you do not have already the dataset downloaded, it will download it automatically inside the `data` directory. The training process will be logged in the `outputs` directory, where you can find the trained model, logs, and other artifacts.

> [!TIP]
> If you want to have interactive view of the training process, you can start tensorboard dashboard with the command `tensorboard --logdir outputs/` and then open your browser at `http://localhost:6006/`. This will allow you to visualize the training process, including loss curves, model parameters, and other metrics.

## Contributing

## Authors and acknowledgment

## License

The project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Project status

The project is currently in active development.
