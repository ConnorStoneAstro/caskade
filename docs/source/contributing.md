# Contributing

Here we explain how you can help improve `caskade`! Perhaps you noticed an odd behaviour/bug, perhaps there is just a feature you'd like to see available, this is how to make it happen!

## 1 Make an Issue

Before doing any coding yourself, you should create an [issue on the GitHub](https://github.com/ConnorStoneAstro/caskade/issues). To do this, first check that your issue doesn't already exist, then write out as clearly as you can any relevant information. If this is a bug, you can include error traces, screenshots, code snippets, and so on to help us root down the problem. For completeness, its best to also include the `caskade` version you are using. If this is a feature request, you'll need to describe the feature, maybe include some dummy code that shows what it would look like to with the feature, and give a clear description of what such a feature would be used for.

Discussion on the issue will whittle down the problem until we either solve it, or have a clear idea of what an update would look like.

## 2 Doing it Yourself

Lets say the issue has gone to the point that you want to try implementing it yourself. Here is how you get your own framework set up:

1. Fork the repo on [GitHub](https://github.com/ConnorStoneAstro/caskade)
1. Clone your version of the repo to your machine: `git clone git@github.com:YourGitHubName/caskade.git`
1. Move into the `caksade` git repo: `cd caksade`
1. Install the editable version: `pip install -e .`
1. Make a new branch: `git checkout -b mynewbranch`
1. Write up your new feature/fix
1. Write some tests in the `tests/` directory
1. Check the tests run by calling `pytest`
1. Commit the changes (really you should do this often while working): `git commit -m "my note on updates"`
1. Push your changes back to GitHub: `git push origin mynewbranch`
1. Create a `Pull Request` on GitHub pointing to the original `caskade` repo
1. Watch the automated tests run and follow any discussion that happens on the PR!