# Titanic Competition

This is my Kaggle submission code.

## Requirements

You need Python 3.6 or higher version.

## Usage

Download the dataset using the following command.

```
kaggle competitions download -c titanic
unzip titanic.zip && rm titanic.zip
```

Then run `titanic.py` as follows.

```
python titanic.py
```

Finally, submit the result using the following command. Replace the `<submission_message>` by yours.

```
kaggle competitions submit -c titanic -f submission.csv -m <submission_message>
```
