import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.class_definition
class Fixed(type):
    def __new__(cls, name, bases, classdict):
        # Create a new base class
        print(classdict)
        new_base_dict = {
            key: property(lambda _: val)
            for key, val in classdict.items()
            if not key.startswith("_")
        }
        new_class_dict = {
            key: val for key, val in classdict.items()
            if key.startswith("_")
        }
        new_base = type.__new__(type, f"FixedBaseOf{name}", (type,), new_base_dict)
        new_cls = type.__new__(new_base, name, bases, new_class_dict)
        return new_cls


@app.class_definition
class Thing(metaclass=Fixed):
    x: int = 4


@app.cell
def _():
    Thing.x
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
