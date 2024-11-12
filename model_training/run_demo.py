__author__ = 'Brett Allen (brettallen777@gmail.com)'

import argparse
import json

parser = argparse.ArgumentParser(
    prog = 'ProgramName',
    description = 'What the program does',
    epilog = 'Text at the bottom of help'
)

parser.add_argument('filename')                              # Positional argument
parser.add_argument('-c', '--count')                         # Option that takes a value
parser.add_argument('-v', '--verbose', action='store_true')  # On/Off flag

class ModelTracker(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, model in kwargs.get('models', []):
            self[name] = model
        if 'models' in self:
            self.pop('models')

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def append(self, entry: Tuple[str, Any]):
        # Prevent duplicates and keep latest model entries
        name, model = entry
        self[name] = model

    def get_names(self) -> List[str]:
        return list(self.keys())

    def get_models(self) -> List[Any]:
        return list(self.values())

    def save(self, model_dir: str):
        print('Saving Models:')
        for idx, entry in enumerate(self.items()):
            name, model = entry
            outpath = os.path.join(model_dir, f'{name}.pkl')
        
            # Save the model
            with open(outpath, 'wb') as f:
                pickle.dump(model, f, protocol=5)
        
            print(f'  {idx+1:>2}) {name:<45} | Saved to "{outpath}"')

    def load(self, model_dir: str):
        for path in glob(os.path.join(model_dir, '*.pkl')):
            print(path)
            name = os.path.splitext(os.path.basename(path))[0]
            with open(path, 'rb') as f:
                model = pickle.load(f)
            self[name] = model

    def __repr__(self):
        return [(name, model) for name, model in self.items()]

def to_title_case(s: str) -> str:
    return re.sub(r'_+', ' ', s).title()

def predict(*features: List[Any]):
    # Create single sample feature vector
    X = np.array([*features])
    X = X.reshape(1, -1)
    X = pd.DataFrame(X, columns=columns)

    preds = []
    for pipeline in pipelines:
        try:
            pred = pipeline.predict(X)
            pred_str = id_to_label.get(int(pred[0]), 'Unknown')
            preds.append(pred_str)
        except ValueError as e:
            preds.append('Prediction Failed (One or More Invalid Inputs)')

    pinned_pred = preds[0]  # Pinned model's prediction
    dropdown_preds = preds[1:]  # Other model predictions

    return [pinned_pred] + dropdown_preds

def reset_sliders():
    # Return the initial values for each slider
    return initial_values

def make_slider(data: pd.DataFrame, col: str, label: str=None, **kwargs) -> gr.Slider:
    label = label or col
    step = kwargs.get('step', 0.1 if data[col].dtype == 'float' else 1)
    return gr.Slider(label=label, minimum=0, maximum=data[col].max(), value=max(0, data[col].median().astype(data[col].dtype)), step=step, interactive=True)

def main():
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    pinned_model_name = 'XGBoost'
    model_names = [pinned_model_name] + [name for name in loaded_models.get_names() if name != pinned_model_name]
    pipelines = loaded_models.get_models()
    pinned_pipeline = pipelines.pop(loaded_models.get_names().index(pinned_model_name))
    pipelines.insert(0, pinned_pipeline)

    # Calculate initial slider values based on data
    initial_values = [max(0, X[col].median().astype(X[col].dtype)) for col in X.columns]  # Use median values as the initial state

    # Inputs
    sliders = [make_slider(data=X, col=col, label=to_title_case(col)) for col in X.columns]

    # Outputs
    pinned_label = gr.Label(label=to_title_case(model_names[0]).upper())
    dropdown_labels = [gr.Label(label=to_title_case(name).upper()) for name in model_names[1:]]

    # Group sliders into chunks per row
    slider_cols = 5
    slider_rows = [sliders[i:i+slider_cols] for i in range(0, len(sliders), slider_cols)]

    # Interface with Accordion for collapsible section
    with gr.Blocks() as demo:
        gr.Markdown("## Airline Passenger Satisfaction Prediction")

        # Input sliders
        # with gr.Row():
        #     for slider in sliders:
        #         slider.render()
        # Input sliders in rows by group
        for row in slider_rows:
            with gr.Row():
                for slider in row:
                    slider.render()

        # Pinned model output
        pinned_label.render()

        # Collapsible section for other model predictions
        with gr.Accordion("Other Model Predictions", open=False):
            for label in dropdown_labels:
                label.render()

        # Buttons for prediction and reset
        with gr.Row():
            submit_button = gr.Button("Predict")
            # reset_button = gr.Button("Reset")

        # Define the function binding for the submit button
        submit_button.click(
            fn=predict,
            inputs=sliders,
            outputs=[pinned_label] + dropdown_labels
        )

        # Define the function binding for the reset button
        # TODO Fix issue/bug with invalid values for transform after resetting.
        # reset_button.click(
        #     fn=reset_sliders,
        #     inputs=None,
        #     outputs=sliders  # Reset all sliders to their initial values
        # )

    demo.launch(inline=False, share=True)

if __name__ == '__main__':
    main()
