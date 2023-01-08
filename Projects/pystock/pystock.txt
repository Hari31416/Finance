python -m build

twine check dist/*

twine upload --repository forwarder ./dist/*
