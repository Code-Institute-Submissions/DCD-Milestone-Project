{% extends 'layout.html' %}

{% block body %}
    <h1>Edit Video</h1>
    {% from "includes/_formhelpers.html" import render_field %}
    <form method="POST" action="">
        {{ form.hidden_tag() }}
        <div class="form-group">
            {{ render_field(form.title, class_="form-control") }}
        </div>
        <div class="form-group">
            {{ render_field(form.link, class_="form-control") }}
        </div>
        <p><input class="btn btn-primary" type="submit" value="Submit">
    </form>
{% endblock %}


@app.route('/edit_video/<int:id>/', methods=['GET', 'POST'])
@login_required
def edit_video(id):
    data = Videos.query.get(id)
    form = VideoForm(request.form)
    form.title.data = data.title
    form.link.data = data.link
    if request.method == 'POST' and form.validate():
        data.title = request.form["title"]
        data.link = request.form["link"]
        db.session.commit()
        flash("Video updated!", "success")
        return redirect(url_for('dashboard'))
return render_template('edit_video.html', form=form)



{% extends 'layout.html' %}

{% block body %}
    <h1>Dashboard</h1>
    <table class="table table-striped table-hover">
        <thead>
        <tr>
            <th>Title</th>
            <th>Author</th>
            <th></th>
            <th></th>
        </tr>
        </thead>
        <tbody>
        {% for video in videos %}
            <tr>
                <td><a href="/video/{{ video.id }}/">{{ video.title }}</a></td>
                <td>{{ video.author }}</td>
                <td><a class="btn btn-default pull-right" href="/edit_video/{{ video.id }}">Edit</a></td>
                <td><a class="btn btn-danger" href="/delete_video/{{ video.id }}">Delete</a></td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
{% endblock body %}