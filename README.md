This is an experiment with using a GRU classifier, and its in progress.
# Character-level Autocomplete

This repo contains a character-based language model to predict next 3 best characters that can be surfaced in an autocomplete interface.

## Creating submission for Gradescope/Canvas

Ensure you are in the same directory as this README file. In terminal, run the following command:
```
./submit.sh
```
Note: If you run into a "Permission Denied" error, run the following command and then rerun the command above.
```
chmod +x submit.sh
```

## Important notes about this submission

Because this checkpoint runs to spec only, the model doesn't actually require training, we simply save a fake checkpoint to `work/model.checkpoint`.