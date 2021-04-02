"""SIR model by Grant Sanderson (3Blue1Brown), used with kind permission"""
from manim import *
from manimlib import *

GREY_D = GREY
FRAME_HEIGHT = 720; FRAME_WIDTH = 1080

SICKLY_GREEN = "#9BBD37"
COLOR_MAP = {
    "S": BLUE,
    "I": RED,
    "R": GREY_D,
}


def update_time(mob, dt):
    mob.time += dt

# Person class - each person is either susceptible, infected or recovered
# (It is assumed 'dead' counts as recovered, and immunity is permanent to simplify)
class Person(VGroup):
    CONFIG = {
        "status": "S",  # S, I or R
        "height": 0.2,
        "color_map": COLOR_MAP,
        "infection_ring_style": {
            "stroke_color": RED,
            "stroke_opacity": 0.8,
            "stroke_width": 0,
        },
        "infection_radius": 0.5,
        "infection_animation_period": 2,
        "symptomatic": False,
        "p_symptomatic_on_infection": 1,
        "max_speed": 1,
        "dl_bound": [-FRAME_WIDTH / 2, -FRAME_HEIGHT / 2],
        "ur_bound": [FRAME_WIDTH / 2, FRAME_HEIGHT / 2],
        "gravity_well": None,
        "gravity_strength": 1,
        "wall_buffer": 1,
        "wander_step_size": 1,
        "wander_step_duration": 1,
        "social_distance_factor": 0,
        "social_distance_color_threshold": 2,
        "n_repulsion_points": 10,
        "social_distance_color": YELLOW,
        "max_social_distance_stroke_width": 5,
        "asymptomatic_color": YELLOW,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.time = 0
        self.last_step_change = -1
        self.change_anims = []
        self.velocity = np.zeros(3)
        self.infection_start_time = np.inf
        self.infection_end_time = np.inf
        self.repulsion_points = []
        self.num_infected = 0

        self.center_point = VectorizedPoint()
        self.add(self.center_point)
        self.add_body()
        self.add_infection_ring()
        self.set_status(self.status, run_time=0)

        # Updaters
        self.add_updater(update_time)
        self.add_updater(lambda m, dt: m.update_position(dt))
        self.add_updater(lambda m, dt: m.update_infection_ring(dt))
        self.add_updater(lambda m: m.progress_through_change_anims())

    def add_body(self):
        body = self.get_body()
        body.set_height(self.height)
        body.move_to(self.get_center())
        self.add(body)
        self.body = body

    def get_body(self, status):
        person = SVGMobject(file_name="person")
        person.set_stroke(width=0)
        return person

    def set_status(self, status, run_time=1):
        start_color = self.color_map[self.status]
        end_color = self.color_map[status]

        if status == "I":
            self.infection_start_time = self.time
            self.infection_ring.set_stroke(width=0, opacity=0)
            if random.random() < self.p_symptomatic_on_infection:
                self.symptomatic = True
            else:
                self.infection_ring.set_color(self.asymptomatic_color)
                end_color = self.asymptomatic_color
        if self.status == "I":
            self.infection_end_time = self.time
            self.symptomatic = False

        anims = [
            UpdateFromAlphaFunc(
                self.body,
                lambda m, a: m.set_color(interpolate_color(
                    start_color, end_color, a
                )),
                run_time=run_time,
            )
        ]
        for anim in anims:
            self.push_anim(anim)

        self.status = status

    def push_anim(self, anim):
        anim.suspend_mobject_updating = False
        anim.begin()
        anim.start_time = self.time
        self.change_anims.append(anim)
        return self

    def pop_anim(self, anim):
        anim.update(1)
        anim.finish()
        self.change_anims.remove(anim)

    def add_infection_ring(self):
        self.infection_ring = Circle(
            radius=self.height / 2,
        )
        self.infection_ring.set_style(**self.infection_ring_style)
        self.add(self.infection_ring)
        self.infection_ring.time = 0
        return self

    def update_position(self, dt):
        center = self.get_center()
        total_force = np.zeros(3)

        # Gravity
        if self.wander_step_size != 0:
            if (self.time - self.last_step_change) > self.wander_step_duration:
                vect = rotate_vector(RIGHT, TAU * random.random())
                self.gravity_well = center + self.wander_step_size * vect
                self.last_step_change = self.time

        if self.gravity_well is not None:
            to_well = (self.gravity_well - center)
            dist = get_norm(to_well)
            if dist != 0:
                total_force += self.gravity_strength * to_well / (dist**3)

        # Potentially avoid neighbors
        if self.social_distance_factor > 0:
            repulsion_force = np.zeros(3)
            min_dist = np.inf
            for point in self.repulsion_points:
                to_point = point - center
                dist = get_norm(to_point)
                if 0 < dist < min_dist:
                    min_dist = dist
                if dist > 0:
                    repulsion_force -= self.social_distance_factor * to_point / (dist**3)
            sdct = self.social_distance_color_threshold
            self.body.set_stroke(
                self.social_distance_color,
                width=clip(
                    (sdct / min_dist) - sdct,
                    # 2 * (sdct / min_dist),
                    0,
                    self.max_social_distance_stroke_width
                ),
                background=True,
            )
            total_force += repulsion_force

        # Avoid walls
        wall_force = np.zeros(3)
        for i in range(2):
            to_lower = center[i] - self.dl_bound[i]
            to_upper = self.ur_bound[i] - center[i]

            # Bounce
            if to_lower < 0:
                self.velocity[i] = abs(self.velocity[i])
                self.set_coord(self.dl_bound[i], i)
            if to_upper < 0:
                self.velocity[i] = -abs(self.velocity[i])
                self.set_coord(self.ur_bound[i], i)

            # Repelling force
            wall_force[i] += max((-1 / self.wall_buffer + 1 / to_lower), 0)
            wall_force[i] -= max((-1 / self.wall_buffer + 1 / to_upper), 0)
        total_force += wall_force

        # Apply force
        self.velocity += total_force * dt

        # Limit speed
        speed = get_norm(self.velocity)
        if speed > self.max_speed:
            self.velocity *= self.max_speed / speed

        # Update position
        self.shift(self.velocity * dt)

    def update_infection_ring(self, dt):
        ring = self.infection_ring
        if not (self.infection_start_time <= self.time <= self.infection_end_time + 1):
            return self

        ring_time = self.time - self.infection_start_time
        period = self.infection_animation_period

        alpha = (ring_time % period) / period
        ring.set_height(interpolate(
            self.height,
            self.infection_radius,
            smooth(alpha),
        ))
        ring.set_stroke(
            width=interpolate(
                0, 5,
                there_and_back(alpha),
            ),
            opacity=min([
                min([ring_time, 1]),
                min([self.infection_end_time + 1 - self.time, 1]),
            ]),
        )

        return self

    def progress_through_change_anims(self):
        for anim in self.change_anims:
            if anim.run_time == 0:
                alpha = 1
            else:
                alpha = (self.time - anim.start_time) / anim.run_time
            anim.interpolate(alpha)
            if alpha >= 1:
                self.pop_anim(anim)

    def get_center(self):
        return self.center_point.get_points()[0]


class DotPerson(Person):
    def get_body(self):
        return Dot()


class PiPerson(Person):
    CONFIG = {
        "mode_map": {
            "S": "guilty",
            "I": "sick",
            "R": "tease",
        }
    }

    def get_body(self):
        return Randolph()

    def set_status(self, status, run_time=1):
        super().set_status(status)

        target = self.body.copy()
        target.change(self.mode_map[status])
        target.set_color(self.color_map[status])

        transform = Transform(self.body, target)
        transform.begin()

        def update(body, alpha):
            transform.update(alpha)
            body.move_to(self.center_point)

        anims = [
            UpdateFromAlphaFunc(self.body, update, run_time=run_time),
        ]
        for anim in anims:
            self.push_anim(anim)

        return self

# VGroup is a collection of "vectorized moving objects" from manim
# might be difficult to decouple animation from the simulation itself
# but with manim (or manimlib) imported, we are able to run and collect data
class SIRSimulation(VGroup):
    CONFIG = {
        "n_cities": 1,
        "city_population": 10,
        "box_size": 7,
        "person_type": PiPerson,
        "person_config": {
            "height": 0.2,
            "infection_radius": 0.6,
            "gravity_strength": 1,
            "wander_step_size": 1,
        },
        "p_infection_per_day": 0.2,
        "infection_time": 5,
        "travel_rate": 0,
        "limit_social_distancing_to_infectious": False,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time = 0
        self.latents = {}; self.latent_step = 0
        self.add_updater(update_time)

        self.add_boxes()
        self.add_people()

        self.add_updater(lambda m, dt: m.update_statusses(dt))
        self.add_updater(lambda m, dt: m.update_latent_variables(dt))

    def add_boxes(self):
        boxes = VGroup()
        for x in range(self.n_cities):
            box = Square()
            box.set_height(self.box_size)
            box.set_stroke(WHITE, 3)
            boxes.add(box)
        boxes.arrange_in_grid(buff=LARGE_BUFF)
        self.add(boxes)
        self.boxes = boxes

    def add_people(self):
        people = VGroup()
        for box in self.boxes:
            dl_bound = box.get_corner(DL)
            ur_bound = box.get_corner(UR)
            box.people = VGroup()
            for x in range(self.city_population):
                person = self.person_type(
                    dl_bound=dl_bound,
                    ur_bound=ur_bound,
                    **self.person_config
                )
                person.move_to([
                    interpolate(lower, upper, random.random())
                    for lower, upper in zip(dl_bound, ur_bound)
                ])
                person.box = box
                box.people.add(person)
                people.add(person)

        # Choose a patient zero
        random.choice(people).set_status("I")
        self.add(people)
        self.people = people

    def update_latent_variables(self, dt):
        self.people_latents = []
        for box in self.boxes:
            for person in box.people:
                self.people_latents.append(person.status)
                c = person.get_center()
                for i in [0,1,2]:
                    self.people_latents.append(c[i])
        self.latents[self.latent_step] = self.people_latents
        self.latent_step += 1

    def update_statusses(self, dt):
        for box in self.boxes:
            s_group, i_group = [
                list(filter(
                    lambda m: m.status == status,
                    box.people
                ))
                for status in ["S", "I"]
            ]
            for s_person in s_group:
                for i_person in i_group:
                    dist = get_norm(i_person.get_center() - s_person.get_center())
                    if dist < s_person.infection_radius and random.random() < self.p_infection_per_day * dt:
                        s_person.set_status("I")
                        i_person.num_infected += 1
            for i_person in i_group:
                if (i_person.time - i_person.infection_start_time) > self.infection_time:
                    i_person.set_status("R")

        # Travel
        if self.travel_rate > 0:
            path_func = path_along_arc(45 * DEGREES)
            for person in self.people:
                if random.random() < self.travel_rate * dt:
                    new_box = random.choice(self.boxes)
                    person.box.people.remove(person)
                    new_box.people.add(person)
                    person.box = new_box
                    person.dl_bound = new_box.get_corner(DL)
                    person.ur_bound = new_box.get_corner(UR)

                    person.old_center = person.get_center()
                    person.new_center = new_box.get_center()
                    anim = UpdateFromAlphaFunc(
                        person,
                        lambda m, a: m.move_to(path_func(
                            m.old_center, m.new_center, a,
                        )),
                        run_time=1,
                    )
                    person.push_anim(anim)

        # Social distancing
        centers = np.array([person.get_center() for person in self.people])
        if self.limit_social_distancing_to_infectious:
            repelled_centers = np.array([
                person.get_center()
                for person in self.people
                if person.symptomatic
            ])
        else:
            repelled_centers = centers

        if len(repelled_centers) > 0:
            for center, person in zip(centers, self.people):
                if person.social_distance_factor > 0:
                    diffs = np.linalg.norm(repelled_centers - center, axis=1)
                    person.repulsion_points = repelled_centers[np.argsort(diffs)[1:person.n_repulsion_points + 1]]

    def get_status_counts(self):
        return np.array([
            len(list(filter(
                lambda m: m.status == status,
                self.people
            )))
            for status in "SIR"
        ])

    def get_status_proportions(self):
        counts = self.get_status_counts()
        return counts / sum(counts)


class SIRGraph(VGroup):
    CONFIG = {
        "color_map": COLOR_MAP,
        "height": 7,
        "width": 5,
        "update_frequency": 0.5,
        "include_braces": False,
    }

    def __init__(self, simulation, **kwargs):
        super().__init__(**kwargs)
        self.simulation = simulation
        self.data = [simulation.get_status_proportions()] * 2
        self.add_axes()
        self.add_graph()
        self.add_x_labels()

        self.time = 0
        self.last_update_time = 0
        self.add_updater(update_time)
        self.add_updater(lambda m: m.update_graph())
        self.add_updater(lambda m: m.update_x_labels())

    def add_axes(self):
        axes = Axes(
            y_min=0,
            y_max=1,
            y_axis_config={
                "tick_frequency": 0.1,
            },
            x_min=0,
            x_max=1,
            axis_config={
                "include_tip": False,
            },
        )
        origin = axes.c2p(0, 0)
        axes.x_axis.set_width(self.width, about_point=origin, stretch=True)
        axes.y_axis.set_height(self.height, about_point=origin, stretch=True)

        self.add(axes)
        self.axes = axes

    def add_graph(self):
        self.graph = self.get_graph(self.data)
        self.add(self.graph)

    def add_x_labels(self):
        self.x_labels = VGroup()
        self.x_ticks = VGroup()
        self.add(self.x_ticks, self.x_labels)

    def get_graph(self, data):
        axes = self.axes
        i_points = []
        s_points = []
        for x, props in zip(np.linspace(0, 1, len(data)), data):
            i_point = axes.c2p(x, props[1])
            s_point = axes.c2p(x, sum(props[:2]))
            i_points.append(i_point)
            s_points.append(s_point)

        r_points = [
            axes.c2p(0, 1),
            axes.c2p(1, 1),
            *s_points[::-1],
            axes.c2p(0, 1),
        ]
        s_points.extend([
            *i_points[::-1],
            s_points[0],
        ])
        i_points.extend([
            axes.c2p(1, 0),
            axes.c2p(0, 0),
            i_points[0],
        ])

        points_lists = [s_points, i_points, r_points]
        regions = VGroup(VMobject(), VMobject(), VMobject())

        for region, status, points in zip(regions, "SIR", points_lists):
            region.set_points_as_corners(points)
            region.set_stroke(width=0)
            region.set_fill(self.color_map[status], 1)
        regions[0].set_fill(opacity=0.5)

        return regions

    def update_graph(self):
        if (self.time - self.last_update_time) > self.update_frequency:
            self.data.append(self.simulation.get_status_proportions())
            self.graph.become(self.get_graph(self.data))
            self.last_update_time = self.time

    def update_x_labels(self):
        tick_height = 0.03 * self.graph.get_height()
        tick_template = Line(DOWN, UP)
        tick_template.set_height(tick_height)

        def get_tick(x):
            tick = tick_template.copy()
            tick.move_to(self.axes.c2p(x / self.time, 0))
            return tick

        def get_label(x, tick):
            label = Integer(x)
            label.set_height(tick_height)
            label.next_to(tick, DOWN, buff=0.5 * tick_height)
            return label

        self.x_labels.set_submobjects([])
        self.x_ticks.set_submobjects([])

        if self.time < 15:
            tick_range = range(1, int(self.time) + 1)
        elif self.time < 50:
            tick_range = range(5, int(self.time) + 1, 5)
        elif self.time < 100:
            tick_range = range(10, int(self.time) + 1, 10)
        else:
            tick_range = range(20, int(self.time) + 1, 20)

        for x in tick_range:
            tick = get_tick(x)
            label = get_label(x, tick)
            self.x_ticks.add(tick)
            self.x_labels.add(label)

        # TODO, if I care, refactor
        if 10 < self.time < 15:
            alpha = (self.time - 10) / 5
            for tick, label in zip(self.x_ticks, self.x_labels):
                if label.get_value() % 5 != 0:
                    label.set_opacity(1 - alpha)
                    tick.set_opacity(1 - alpha)
        if 45 < self.time < 50:
            alpha = (self.time - 45) / 5
            for tick, label in zip(self.x_ticks, self.x_labels):
                if label.get_value() % 10 == 5:
                    label.set_opacity(1 - alpha)
                    tick.set_opacity(1 - alpha)

    def add_v_line(self, line_time=None, color=YELLOW, stroke_width=3):
        if line_time is None:
            line_time = self.time

        axes = self.axes
        v_line = Line(
            axes.c2p(1, 0), axes.c2p(1, 1),
            stroke_color=color,
            stroke_width=stroke_width,
        )
        v_line.add_updater(
            lambda m: m.move_to(
                axes.c2p(line_time / max(self.time, 1e-6), 0),
                DOWN,
            )
        )

        self.add(v_line)


class GraphBraces(VGroup):
    CONFIG = {
        "update_frequency": 0.5,
    }

    def __init__(self, graph, simulation, **kwargs):
        super().__init__(**kwargs)
        axes = self.axes = graph.axes
        self.simulation = simulation

        ys = np.linspace(0, 1, 4)
        self.lines = VGroup(*[
            Line(axes.c2p(1, y1), axes.c2p(1, y2))
            for y1, y2 in zip(ys, ys[1:])
        ])
        self.braces = VGroup(*[Brace(line, RIGHT) for line in self.lines])
        self.labels = VGroup(
            TexText("Susceptible", color=COLOR_MAP["S"]),
            TexText("Infectious", color=COLOR_MAP["I"]),
            TexText("Removed", color=COLOR_MAP["R"]),
        )

        self.max_label_height = graph.get_height() * 0.05

        self.add(self.braces, self.labels)

        self.time = 0
        self.last_update_time = -1
        self.add_updater(update_time)
        self.add_updater(lambda m: m.update_braces())
        self.update(0)

    def update_braces(self):
        if (self.time - self.last_update_time) <= self.update_frequency:
            return

        self.last_update_time = self.time
        lines = self.lines
        braces = self.braces
        labels = self.labels
        axes = self.axes

        props = self.simulation.get_status_proportions()
        ys = np.cumsum([0, props[1], props[0], props[2]])

        epsilon = 1e-6
        for i, y1, y2 in zip([1, 0, 2], ys, ys[1:]):
            lines[i].set_points_as_corners([
                axes.c2p(1, y1),
                axes.c2p(1, y2),
            ])
            height = lines[i].get_height()

            braces[i].set_height(
                max(height, epsilon),
                stretch=True
            )
            braces[i].next_to(lines[i], RIGHT)
            label_height = clip(height, epsilon, self.max_label_height)
            labels[i].scale(label_height / labels[i][0][0].get_height())
            labels[i].next_to(braces[i], RIGHT)
        return self


class ValueSlider(NumberLine):
    CONFIG = {
        "x_min": 0,
        "x_max": 1,
        "tick_frequency": 0.1,
        "numbers_with_elongated_ticks": [],
        "numbers_to_show": np.linspace(0, 1, 6),
        "decimal_number_config": {
            "num_decimal_places": 1,
        },
        "stroke_width": 5,
        "width": 8,
        "marker_color": BLUE,
    }

    def __init__(self, name, value, **kwargs):
        super().__init__(**kwargs)
        self.set_width(self.width, stretch=True)
        self.add_numbers()

        self.marker = ArrowTip(start_angle=-90 * DEGREES)
        self.marker.move_to(self.n2p(value), DOWN)
        self.marker.set_color(self.marker_color)
        self.add(self.marker)

        # self.label = DecimalNumber(value)
        # self.label.next_to(self.marker, UP)
        # self.add(self.label)

        self.name = TexText(name)
        self.name.scale(1.25)
        self.name.next_to(self, DOWN)
        self.name.match_color(self.marker)
        self.add(self.name)

    def get_change_anim(self, new_value, **kwargs):
        start_value = self.p2n(self.marker.get_bottom())
        # m2l = self.label.get_center() - self.marker.get_center()

        def update(mob, alpha):
            interim_value = interpolate(start_value, new_value, alpha)
            mob.marker.move_to(mob.n2p(interim_value), DOWN)
            # mob.label.move_to(mob.marker.get_center() + m2l)
            # mob.label.set_value(interim_value)

        return UpdateFromAlphaFunc(self, update, **kwargs)


# Scenes

class Test(Scene):
    def construct(self):
        path_func = path_along_arc(45 * DEGREES)
        person = PiPerson(height=1, gravity_strength=0.2)
        person.gravity_strength = 0

        person.old_center = person.get_center()
        person.new_center = 4 * RIGHT

        self.add(person)
        self.wait()

        self.play(UpdateFromAlphaFunc(
            person,
            lambda m, a: m.move_to(path_func(
                m.old_center,
                m.new_center,
                a,
            )),
            run_time=3,
            rate_func=there_and_back,
        ))

        self.wait(3)
        self.wait(3)


class RunSimpleSimulation(Scene):
    CONFIG = {
        "simulation_config": {
            "person_type": PiPerson,
            "n_cities": 1,
            "city_population": 10,
            "person_config": {
                "infection_radius": 0.75,
                "social_distance_factor": 0,
                "gravity_strength": 0.2,
                "max_speed": 0.5,
            },
            "travel_rate": 0,
            "infection_time": 5,
        },
        "graph_config": {
            "update_frequency": 1 / 15,
        },
        "graph_height_to_frame_height": 0.5,
        "graph_width_to_frame_height": 0.75,
        "include_graph_braces": True,
    }

    def setup(self):
        self.add_simulation()
        self.position_camera()
        self.add_graph()
        self.add_sliders()
        self.add_R_label()
        self.add_total_cases_label()

    def construct(self):
        self.latentData = self.run_until_zero_infections(print_latents=False)
        return self.latentData

    def wait_until_infection_threshold(self, threshold):
        self.wait_until(lambda: self.simulation.get_status_counts()[1] > threshold)

    def run_until_zero_infections(self, print_latents=False):
        while True:
            self.wait(5)
            if self.simulation.get_status_counts()[1] == 0:
                if (print_latents): print(self.simulation.latents)
                return self.simulation.latents

    def add_R_label(self):
        label = VGroup(
            Tex("R = "),
            DecimalNumber(),
        )
        label.arrange(RIGHT)
        boxes = self.simulation.boxes
        label.set_width(0.25 * boxes.get_width())
        label.next_to(boxes.get_corner(DL), DR)
        self.add(label)

        all_R0_values = []

        def update_label(label):
            if (self.time - label.last_update_time) < label.update_period:
                return
            label.last_update_time = self.time

            values = []
            for person in self.simulation.people:
                if person.status == "I":
                    prop = (person.time - person.infection_start_time) / self.simulation.infection_time
                    if prop > 0.1:
                        values.append(person.num_infected / prop)
            if len(values) > 0:
                all_R0_values.append(np.mean(values))
                average = np.mean(all_R0_values[-20:])
                label[1].set_value(average)

        label.last_update_time = 0
        label.update_period = 1
        label.add_updater(update_label)

    def add_total_cases_label(self):
        label = VGroup(
            TexText("\\# Active cases = "),
            Integer(1),
        )
        label.arrange(RIGHT)
        label[1].align_to(label[0][0][1], DOWN)
        label.set_color(RED)
        boxes = self.simulation.boxes
        label.set_width(0.5 * boxes.get_width())
        label.next_to(boxes, UP, buff=0.03 * boxes.get_width())

        label.add_updater(
            lambda m: m[1].set_value(self.simulation.get_status_counts()[1])
        )
        self.total_cases_label = label
        self.add(label)

    def add_simulation(self):
        self.simulation = SIRSimulation(**self.simulation_config)
        self.add(self.simulation)

    def position_camera(self):
        frame = self.camera.frame
        boxes = self.simulation.boxes
        min_height = boxes.get_height() + 1
        min_width = 3 * boxes.get_width()
        if frame.get_height() < min_height:
            frame.set_height(min_height)
        if frame.get_width() < min_width:
            frame.set_width(min_width)

        frame.next_to(boxes.get_right(), LEFT, buff=-0.1 * boxes.get_width())

    def add_graph(self):
        frame = self.camera.frame
        frame_height = frame.get_height()
        graph = SIRGraph(
            self.simulation,
            height=self.graph_height_to_frame_height * frame_height,
            width=self.graph_width_to_frame_height * frame_height,
            **self.graph_config,
        )
        graph.move_to(frame, UL)
        graph.shift(0.05 * DR * frame_height)
        self.add(graph)
        self.graph = graph

        if self.include_graph_braces:
            self.graph_braces = GraphBraces(
                graph,
                self.simulation,
                update_frequency=graph.update_frequency
            )
            self.add(self.graph_braces)

    def add_sliders(self):
        pass


class RunSimpleSimulationWithDots(RunSimpleSimulation):
    CONFIG = {
        "simulation_config": {
            "person_type": DotPerson,
        }
    }

class LargerCity(RunSimpleSimulation):
    CONFIG = {
        "simulation_config": {
            "person_type": DotPerson,
            "city_population": 1000,
            "person_config": {
                "infection_radius": 0.25,
                "social_distance_factor": 0,
                "gravity_strength": 0.2,
                "max_speed": 0.25,
                "height": 0.2 / 3,
                "wall_buffer": 1 / 3,
                "social_distance_color_threshold": 2 / 3,
            },
        }
    }


class LargerCity2(LargerCity):
    CONFIG = {
        "random_seed": 1,
    }


class LargeCityHighInfectionRadius(LargerCity):
    CONFIG = {
        "simulation_config": {
            "person_config": {
                "infection_radius": 0.5,
            },
        },
        "graph_config": {
            "update_frequency": 1 / 60,
        },
    }


class LargeCityLowerInfectionRate(LargerCity):
    CONFIG = {
        "p_infection_per_day": 0.1,
    }


class SimpleSocialDistancing(RunSimpleSimulation):
    CONFIG = {
        "simulation_config": {
            "person_type": PiPerson,
            "n_cities": 1,
            "city_population": 100,
            "person_config": {
                "infection_radius": 0.75,
                "social_distance_factor": 2,
                "gravity_strength": 0.1,
            },
            "travel_rate": 0,
            "infection_time": 5,
        },
    }


class DelayedSocialDistancing(RunSimpleSimulation):
    CONFIG = {
        "delay_time": 8,
        "target_sd_factor": 2,
        "sd_probability": 1,
        "random_seed": 1,
    }

    def construct(self):
        self.wait(self.delay_time)
        self.change_social_distance_factor(
            self.target_sd_factor,
            self.sd_probability,
        )
        self.graph.add_v_line()
        self.play(self.sd_slider.get_change_anim(self.target_sd_factor))

        self.run_until_zero_infections()

    def change_social_distance_factor(self, new_factor, prob):
        for person in self.simulation.people:
            if random.random() < prob:
                person.social_distance_factor = new_factor

    def add_sliders(self):
        slider = ValueSlider(
            self.get_sd_slider_name(),
            value=0,
            x_min=0,
            x_max=2,
            tick_frequency=0.5,
            numbers_with_elongated_ticks=[],
            numbers_to_show=range(3),
            decimal_number_config={
                "num_decimal_places": 0,
            }
        )
        slider.match_width(self.graph)
        slider.next_to(self.graph, DOWN, buff=0.2 * self.graph.get_height())
        self.add(slider)
        self.sd_slider = slider

    def get_sd_slider_name(self):
        return f"Social Distance Factor\\\\({int(100 * self.sd_probability)}$\\%$ of population)"


class DelayedSocialDistancingDot(DelayedSocialDistancing):
    CONFIG = {
        "simulation_config": {
            "person_type": DotPerson,
        }
    }


class DelayedSocialDistancingLargeCity(DelayedSocialDistancing, LargerCity):
    CONFIG = {
        "trigger_infection_count": 50,
        "simulation_config": {
            'city_population': 900,
        }
    }

    def construct(self):
        self.wait_until_infection_threshold(self.trigger_infection_count)
        self.change_social_distance_factor(
            self.target_sd_factor,
            self.sd_probability,
        )
        self.graph.add_v_line()
        self.play(self.sd_slider.get_change_anim(self.target_sd_factor))

        self.run_until_zero_infections()


class DelayedSocialDistancingLargeCity90p(DelayedSocialDistancingLargeCity):
    CONFIG = {
        "sd_probability": 0.9,
    }


class DelayedSocialDistancingLargeCity90pAlt(DelayedSocialDistancingLargeCity):
    CONFIG = {
        "sd_probability": 0.9,
        "random_seed": 5,
    }


class DelayedSocialDistancingLargeCity70p(DelayedSocialDistancingLargeCity):
    CONFIG = {
        "sd_probability": 0.7,
    }


class DelayedSocialDistancingLargeCity50p(DelayedSocialDistancingLargeCity):
    CONFIG = {
        "sd_probability": 0.5,
    }


class DelayedSocialDistancingWithDots(DelayedSocialDistancing):
    CONFIG = {
        "person_type": DotPerson,
    }


class DelayedSocialDistancingProbHalf(DelayedSocialDistancing):
    CONFIG = {
        "sd_probability": 0.5,
    }


class ReduceInfectionDuration(LargerCity):
    CONFIG = {
        "trigger_infection_count": 50,
    }

    def construct(self):
        self.wait_until_infection_threshold(self.trigger_infection_count)
        self.play(self.slider.get_change_anim(1))
        self.simulation.infection_time = 1
        self.graph.add_v_line()
        self.run_until_zero_infections()

    def add_sliders(self):
        slider = ValueSlider(
            "Infection duration",
            value=5,
            x_min=0,
            x_max=5,
            tick_frequency=1,
            numbers_with_elongated_ticks=[],
            numbers_to_show=range(6),
            decimal_number_config={
                "num_decimal_places": 0,
            },
            marker_color=RED,
        )
        slider.match_width(self.graph)
        slider.next_to(self.graph, DOWN, buff=0.2 * self.graph.get_height())
        self.add(slider)
        self.slider = slider


class SimpleTravel(RunSimpleSimulation):
    CONFIG = {
        "simulation_config": {
            "person_type": DotPerson,
            "n_cities": 12,
            "city_population": 100,
            "person_config": {
                "infection_radius": 0.75,
                "social_distance_factor": 0,
                "gravity_strength": 0.5,
            },
            "travel_rate": 0.02,
            "infection_time": 5,
        },
    }

    def add_sliders(self):
        slider = ValueSlider(
            "Travel rate",
            self.simulation.travel_rate,
            x_min=0,
            x_max=0.02,
            tick_frequency=0.005,
            numbers_with_elongated_ticks=[],
            numbers_to_show=np.arange(0, 0.03, 0.01),
            decimal_number_config={
                "num_decimal_places": 2,
            }
        )
        slider.match_width(self.graph)
        slider.next_to(self.graph, DOWN, buff=0.2 * self.graph.get_height())
        self.add(slider)
        self.tr_slider = slider


class SimpleTravel2(SimpleTravel):
    CONFIG = {
        "random_seed": 1,
    }


class SimpleTravelLongInfectionPeriod(SimpleTravel):
    CONFIG = {
        "simulation_config": {
            "infection_time": 10,
        }
    }


class SimpleTravelDelayedSocialDistancing(DelayedSocialDistancing, SimpleTravel):
    CONFIG = {
        "target_sd_factor": 2,
        "sd_probability": 0.7,
        "delay_time": 15,
        "trigger_infection_count": 50,
    }

    def construct(self):
        self.wait_until_infection_threshold(self.trigger_infection_count)
        self.change_social_distance_factor(
            self.target_sd_factor,
            self.sd_probability,
        )
        self.graph.add_v_line()
        self.play(self.sd_slider.get_change_anim(self.target_sd_factor))

        self.run_until_zero_infections()

    def add_sliders(self):
        SimpleTravel.add_sliders(self)
        DelayedSocialDistancing.add_sliders(self)

        buff = 0.1 * self.graph.get_height()

        self.tr_slider.scale(0.8, about_edge=UP)
        self.tr_slider.next_to(self.graph, DOWN, buff=buff)

        self.sd_slider.scale(0.8)
        self.sd_slider.marker.set_color(YELLOW)
        self.sd_slider.name.set_color(YELLOW)
        self.sd_slider.next_to(self.tr_slider, DOWN, buff=buff)


class SimpleTravelDelayedSocialDistancing70p(SimpleTravelDelayedSocialDistancing):
    pass


class SimpleTravelDelayedSocialDistancing99p(SimpleTravelDelayedSocialDistancing):
    CONFIG = {
        "sd_probability": 0.99,
    }


class SimpleTravelDelayedSocialDistancing20p(SimpleTravelDelayedSocialDistancing):
    CONFIG = {
        "sd_probability": 0.20,
    }


class SimpleTravelDelayedSocialDistancing50p(SimpleTravelDelayedSocialDistancing):
    CONFIG = {
        "sd_probability": 0.50,
        "random_seed": 1,
    }


class SimpleTravelDelayedSocialDistancing50pThreshold100(SimpleTravelDelayedSocialDistancing):
    CONFIG = {
        "sd_probability": 0.50,
        "trigger_infection_count": 100,
        "random_seed": 5,
    }


class SimpleTravelDelayedSocialDistancing70pThreshold100(SimpleTravelDelayedSocialDistancing):
    CONFIG = {
        "sd_probability": 0.70,
        "trigger_infection_count": 100,
    }


class SimpleTravelSocialDistancePlusZeroTravel(SimpleTravelDelayedSocialDistancing):
    CONFIG = {
        "sd_probability": 1,
        "target_travel_rate": 0,
    }

    def construct(self):
        self.wait_until_infection_threshold(self.trigger_infection_count)
        self.change_social_distance_factor(
            self.target_sd_factor,
            self.sd_probability,
        )
        self.simulation.travel_rate = self.target_travel_rate
        self.graph.add_v_line()
        self.play(
            self.tr_slider.get_change_anim(self.simulation.travel_rate),
            self.sd_slider.get_change_anim(self.target_sd_factor),
        )

        self.run_until_zero_infections()


class SecondWave(SimpleTravelSocialDistancePlusZeroTravel):
    def run_until_zero_infections(self):
        self.wait_until(lambda: self.simulation.get_status_counts()[1] < 10)
        self.change_social_distance_factor(0, 1)
        self.simulation.travel_rate = 0.02
        self.graph.add_v_line()
        self.play(
            self.tr_slider.get_change_anim(0.02),
            self.sd_slider.get_change_anim(0),
        )
        super().run_until_zero_infections()


class SimpleTravelSocialDistancePlusZeroTravel99p(SimpleTravelSocialDistancePlusZeroTravel):
    CONFIG = {
        "sd_probability": 0.99,
    }


class SimpleTravelDelayedTravelReduction(SimpleTravelDelayedSocialDistancing):
    CONFIG = {
        "trigger_infection_count": 50,
        "target_travel_rate": 0.002,
    }

    def construct(self):
        self.wait_until_infection_threshold(self.trigger_infection_count)
        self.simulation.travel_rate = self.target_travel_rate
        self.graph.add_v_line()
        self.play(self.tr_slider.get_change_anim(self.simulation.travel_rate))
        self.run_until_zero_infections()


class SimpleTravelDelayedTravelReductionThreshold100(SimpleTravelDelayedTravelReduction):
    CONFIG = {
        "random_seed": 2,
        "trigger_infection_count": 100,
    }


class SimpleTravelDelayedTravelReductionThreshold100TargetHalfPercent(SimpleTravelDelayedTravelReduction):
    CONFIG = {
        "random_seed": 2,
        "trigger_infection_count": 100,
        "target_travel_rate": 0.005,
    }


class SimpleTravelDelayedTravelReductionThreshold100TargetHalfPercent2(SimpleTravelDelayedTravelReductionThreshold100TargetHalfPercent):
    CONFIG = {
        "random_seed": 1,
        "sd_probability": 0.5,
    }

    def setup(self):
        super().setup()
        for x in range(2):
            random.choice(self.simulation.people).set_status("I")


class SimpleTravelLargeCity(SimpleTravel, LargerCity):
    CONFIG = {
        "simulation_config": {
            "n_cities": 12,
            "travel_rate": 0.02,
        }
    }


class SimpleTravelLongerDelayedSocialDistancing(SimpleTravelDelayedSocialDistancing):
    CONFIG = {
        "trigger_infection_count": 100,
    }


class SimpleTravelLongerDelayedTravelReduction(SimpleTravelDelayedTravelReduction):
    CONFIG = {
        "trigger_infection_count": 100,
    }


class SocialDistanceAfterFiveDays(DelayedSocialDistancing):
    CONFIG = {
        "sd_probability": 0.7,
        "delay_time": 5,
        "simulation_config": {
            "travel_rate": 0
        },
    }


class QuarantineInfectious(RunSimpleSimulation):
    CONFIG = {
        "trigger_infection_count": 10,
        "target_sd_factor": 3,
        "infection_time_before_quarantine": 1,
    }

    def construct(self):
        self.wait_until_infection_threshold(self.trigger_infection_count)
        self.add_quarantine_box()
        self.set_quarantine_updaters()
        self.run_until_zero_infections()

    def add_quarantine_box(self):
        boxes = self.simulation.boxes
        q_box = boxes[0].copy()
        q_box.set_color(RED_E)
        q_box.set_width(boxes.get_width() / 3)
        q_box.next_to(
            boxes, LEFT,
            aligned_edge=DOWN,
            buff=0.25 * q_box.get_width()
        )

        label = TexText("Quarantine zone")
        label.set_color(RED)
        label.match_width(q_box)
        label.next_to(q_box, DOWN, buff=0.1 * q_box.get_width())

        self.add(q_box)
        self.add(label)
        self.q_box = q_box

    def set_quarantine_updaters(self):
        def quarantine_if_ready(simulation):
            for person in simulation.people:
                send_to_q_box = all([
                    not person.is_quarantined,
                    person.symptomatic,
                    (person.time - person.infection_start_time) > self.infection_time_before_quarantine,
                ])
                if send_to_q_box:
                    person.box = self.q_box
                    person.dl_bound = self.q_box.get_corner(DL)
                    person.ur_bound = self.q_box.get_corner(UR)
                    person.old_center = person.get_center()
                    person.new_center = self.q_box.get_center()
                    point = VectorizedPoint(person.get_center())
                    person.push_anim(ApplyMethod(point.move_to, self.q_box.get_center(), run_time=0.5))
                    person.push_anim(MaintainPositionRelativeTo(person, point))
                    person.move_to(self.q_box.get_center())
                    person.is_quarantined = True

        for person in self.simulation.people:
            person.is_quarantined = False
            # person.add_updater(quarantine_if_ready)
        self.simulation.add_updater(quarantine_if_ready)


class QuarantineInfectiousLarger(QuarantineInfectious, LargerCity):
    CONFIG = {
        "trigger_infection_count": 50,
    }


class QuarantineInfectiousLargerWithTail(QuarantineInfectiousLarger):
    def construct(self):
        super().construct()
        self.simulation.clear_updaters()
        self.wait(25)


class QuarantineInfectiousTravel(QuarantineInfectious, SimpleTravel):
    CONFIG = {
        "trigger_infection_count": 50,
    }

    def add_sliders(self):
        pass


class QuarantineInfectious80p(QuarantineInfectious):
    CONFIG = {
        "simulation_config": {
            "person_config": {
                "p_symptomatic_on_infection": 0.8,
            }
        }
    }


class QuarantineInfectiousLarger80p(QuarantineInfectiousLarger):
    CONFIG = {
        "simulation_config": {
            "person_config": {
                "p_symptomatic_on_infection": 0.8,
            }
        }
    }


class QuarantineInfectiousTravel80p(QuarantineInfectiousTravel):
    CONFIG = {
        "simulation_config": {
            "person_config": {
                "p_symptomatic_on_infection": 0.8,
            }
        }
    }


class QuarantineInfectious50p(QuarantineInfectious):
    CONFIG = {
        "simulation_config": {
            "person_config": {
                "p_symptomatic_on_infection": 0.5,
            }
        }
    }


class QuarantineInfectiousLarger50p(QuarantineInfectiousLarger):
    CONFIG = {
        "simulation_config": {
            "person_config": {
                "p_symptomatic_on_infection": 0.5,
            }
        }
    }


class QuarantineInfectiousTravel50p(QuarantineInfectiousTravel):
    CONFIG = {
        "simulation_config": {
            "person_config": {
                "p_symptomatic_on_infection": 0.5,
            }
        }
    }


class CentralMarket(DelayedSocialDistancing):
    CONFIG = {
        "sd_probability": 0.7,
        "delay_time": 5,
        "simulation_config": {
            "person_type": DotPerson,
            "travel_rate": 0
        },
        "shopping_frequency": 0.05,
        "shopping_time": 1,
    }

    def setup(self):
        super().setup()
        for person in self.simulation.people:
            person.last_shopping_trip = -3
            person.is_shopping = False

        square = Square()
        square.set_height(0.2)
        square.set_color(WHITE)
        square.move_to(self.simulation.boxes[0].get_center())
        self.add(square)

        self.simulation.add_updater(
            lambda m, dt: self.add_travel_anims(m, dt)
        )

    def construct(self):
        self.run_until_zero_infections()

    def add_travel_anims(self, simulation, dt):
        shopping_time = self.shopping_time
        for person in simulation.people:
            time_since_trip = person.time - person.last_shopping_trip
            if time_since_trip > shopping_time:
                if random.random() < dt * self.shopping_frequency:
                    person.last_shopping_trip = person.time

                    point = VectorizedPoint(person.get_center())
                    anim1 = ApplyMethod(
                        point.move_to, person.box.get_center(),
                        path_arc=45 * DEGREES,
                        run_time=shopping_time,
                        rate_func=there_and_back_with_pause,
                    )
                    anim2 = MaintainPositionRelativeTo(person, point, run_time=shopping_time)

                    person.push_anim(anim1)
                    person.push_anim(anim2)

    def add_sliders(self):
        pass


class CentralMarketLargePopulation(CentralMarket, LargerCity):
    pass


class CentralMarketLowerInfection(CentralMarketLargePopulation):
    CONFIG = {
        "simulation_config": {
            "p_infection_per_day": 0.1,
        }
    }


class CentralMarketVeryFrequentLargePopulationDelayedSocialDistancing(CentralMarketLowerInfection):
    CONFIG = {
        "sd_probability": 0.7,
        "trigger_infection_count": 25,
        "simulation_config": {
            "person_type": DotPerson,
        },
        "target_sd_factor": 2,
    }

    def construct(self):
        self.wait_until_infection_threshold(self.trigger_infection_count)
        self.graph.add_v_line()
        for person in self.simulation.people:
            person.social_distance_factor = self.target_sd_factor
        self.run_until_zero_infections()


class CentralMarketLessFrequent(CentralMarketVeryFrequentLargePopulationDelayedSocialDistancing):
    CONFIG = {
        "target_shopping_frequency": 0.01,
        "trigger_infection_count": 100,
        "random_seed": 1,
        "simulation_config": {
            'city_population': 900,
        },
    }

    def construct(self):
        self.wait_until(lambda: self.simulation.get_status_counts()[1] > self.trigger_infection_count)
        for person in self.simulation.people:
            person.social_distance_factor = 2
        # Decrease shopping rate
        self.graph.add_v_line()
        self.change_slider()
        self.run_until_zero_infections()

    def change_slider(self):
        self.play(self.shopping_slider.get_change_anim(self.target_shopping_frequency))
        self.shopping_frequency = self.target_shopping_frequency

    def add_sliders(self):
        slider = ValueSlider(
            "Shopping frequency",
            value=self.shopping_frequency,
            x_min=0,
            x_max=0.05,
            tick_frequency=0.01,
            numbers_with_elongated_ticks=[],
            numbers_to_show=np.arange(0, 0.06, 0.01),
            decimal_number_config={
                "num_decimal_places": 2,
            }
        )
        slider.match_width(self.graph)
        slider.next_to(self.graph, DOWN, buff=0.2 * self.graph.get_height())
        self.add(slider)
        self.shopping_slider = slider


class CentralMarketDownToZeroFrequency(CentralMarketLessFrequent):
    CONFIG = {
        "target_shopping_frequency": 0,
    }


class CentralMarketQuarantine(QuarantineInfectiousLarger, CentralMarketLowerInfection):
    CONFIG = {
        "random_seed": 1,
    }

    def construct(self):
        self.wait_until_infection_threshold(self.trigger_infection_count)
        self.graph.add_v_line()
        self.add_quarantine_box()
        self.set_quarantine_updaters()
        self.run_until_zero_infections()


class CentralMarketQuarantine80p(CentralMarketQuarantine):
    CONFIG = {
        "simulation_config": {
            "person_config": {
                "p_symptomatic_on_infection": 0.8,
            }
        }
    }


class CentralMarketTransitionToLowerInfection(CentralMarketLessFrequent):
    CONFIG = {
        "target_p_infection_per_day": 0.05,  # From 0.1
        "trigger_infection_count": 100,
        "random_seed": 1,
        "simulation_config": {
            'city_population': 900,
        },
    }

    def change_slider(self):
        self.play(self.infection_slider.get_change_anim(self.target_p_infection_per_day))
        self.simulation.p_infection_per_day = self.target_p_infection_per_day

    def add_sliders(self):
        slider = ValueSlider(
            "Infection rate",
            value=self.simulation.p_infection_per_day,
            x_min=0,
            x_max=0.2,
            tick_frequency=0.05,
            numbers_with_elongated_ticks=[],
            numbers_to_show=np.arange(0, 0.25, 0.05),
            decimal_number_config={
                "num_decimal_places": 2,
            },
            marker_color=RED,
        )
        slider.match_width(self.graph)
        slider.next_to(self.graph, DOWN, buff=0.2 * self.graph.get_height())
        self.add(slider)
        self.infection_slider = slider


class CentralMarketTransitionToLowerInfectionAndLowerFrequency(CentralMarketTransitionToLowerInfection):
    CONFIG = {
        "random_seed": 2,
    }

    def change_slider(self):
        super().change_slider()
        self.shopping_frequency = self.target_shopping_frequency


class DescribeModel(Scene):
    def construct(self):
        # Setup words
        words = TexText(
            "Susceptible",
            "Infectious",
            "Recovered",
        )
        words.scale(0.8 / words[0][0].get_height())

        colors = [
            COLOR_MAP["S"],
            COLOR_MAP["I"],
            interpolate_color(COLOR_MAP["R"], WHITE, 0.5),
        ]

        initials = VGroup()
        for i, word, color in zip(it.count(), words, colors):
            initials.add(word[0][0])
            word.set_color(color)
            word.move_to(2 * i * DOWN)
            word.to_edge(LEFT)
        words.to_corner(UL)

        # Rearrange initials
        initials.save_state()
        initials.arrange(RIGHT, buff=SMALL_BUFF)
        initials.set_color(WHITE)

        title = VGroup(initials, TexText("Model"))
        title[1].match_height(title[0])
        title.arrange(RIGHT, buff=MED_LARGE_BUFF)
        title.center()
        title.to_edge(UP)

        self.play(FadeInFromDown(title))
        self.wait()
        self.play(
            Restore(
                initials,
                path_arc=-90 * DEGREES,
            ),
            FadeOut(title[1])
        )
        self.wait()

        # Show each category
        pi = PiPerson(
            height=3,
            max_speed=0,
            infection_radius=5,
        )
        pi.color_map["R"] = words[2].get_color()
        pi.center()
        pi.body.change("pondering", initials[0])

        word_anims = []
        for word in words:
            word_anims.append(LaggedStartMap(
                FadeInFrom, word[1:],
                lambda m: (m, 0.2 * LEFT),
            ))

        self.play(
            Succession(
                FadeInFromDown(pi),
                ApplyMethod(pi.body.change, "guilty"),
            ),
            word_anims[0],
            run_time=2,
        )
        words[0].pi = pi.copy()
        self.play(
            words[0].pi.set_height, 1,
            words[0].pi.next_to, words[0], RIGHT,
        )
        self.play(Blink(pi.body))

        pi.set_status("I")
        point = VectorizedPoint(pi.get_center())
        self.play(
            point.shift, 3 * RIGHT,
            MaintainPositionRelativeTo(pi, point),
            word_anims[1],
            run_time=2,
        )
        words[1].pi = pi.copy()
        self.play(
            words[1].pi.set_height, 1,
            words[1].pi.next_to, words[1], RIGHT,
        )
        self.wait(3)

        pi.set_status("R")
        self.play(
            word_anims[2],
            Animation(pi, suspend_mobject_updating=False)
        )
        words[2].pi = pi.copy()
        self.play(
            words[2].pi.set_height, 1,
            words[2].pi.next_to, words[2], RIGHT,
        )
        self.wait()

        # Show rules
        i_pi = PiPerson(
            height=1.5,
            max_speed=0,
            infection_radius=6,
            status="S",
        )
        i_pi.set_status("I")
        s_pis = VGroup()
        for vect in [RIGHT, UP, LEFT, DOWN]:
            s_pi = PiPerson(
                height=1.5,
                max_speed=0,
                infection_radius=6,
                status="S",
            )
            s_pi.next_to(i_pi, vect, MED_LARGE_BUFF)
            s_pis.add(s_pi)

        VGroup(i_pi, s_pis).to_edge(RIGHT)

        circle = Circle(radius=3)
        circle.move_to(i_pi)
        dashed_circle = DashedVMobject(circle, num_dashes=30)
        dashed_circle.set_color(RED)

        self.play(
            FadeOut(pi),
            FadeIn(s_pis),
            FadeIn(i_pi),
        )
        anims = []
        for s_pi in s_pis:
            anims.append(ApplyMethod(s_pi.body.look_at, i_pi.body.eyes))
        self.play(*anims)
        self.add(VGroup(i_pi, *s_pis))
        self.wait()
        self.play(ShowCreation(dashed_circle))
        self.wait()
        shuffled = list(s_pis)
        random.shuffle(shuffled)
        for s_pi in shuffled:
            s_pi.set_status("I")
            self.wait(3 * random.random())
        self.wait(2)
        self.play(FadeOut(s_pis), FadeOut(dashed_circle))

        # Let time pass
        clock = Clock()
        clock.next_to(i_pi.body, UP, buff=LARGE_BUFF)

        self.play(
            VFadeIn(clock),
            ClockPassesTime(
                clock,
                run_time=5,
                hours_passed=5,
            ),
        )
        i_pi.set_status("R")
        self.wait(1)
        self.play(Blink(i_pi.body))
        self.play(FadeOut(clock))

        # Removed
        removed = TexText("Removed")
        removed.match_color(words[2])
        removed.match_height(words[2])
        removed.move_to(words[2], DL)

        self.play(
            FadeOut(words[2], UP),
            FadeIn(removed, DOWN),
        )
        self.play(
            i_pi.body.change, 'pleading', removed,
        )
        self.play(Blink(i_pi.body))
        self.wait()


class DoubleRadiusInGroup(Scene):
    def construct(self):
        radius = 1

        pis = VGroup(*[
            PiPerson(
                height=0.5,
                max_speed=0,
                wander_step_size=0,
                infection_radius=4 * radius,
            )
            for x in range(49)
        ])
        pis.arrange_in_grid()
        pis.set_height(FRAME_HEIGHT - 1)
        sicky = pis[24]
        sicky.set_status("I")

        circle = Circle(radius=radius)
        circle.move_to(sicky)
        dashed_circle = DashedVMobject(circle, num_dashes=30)
        dashed_circle2 = dashed_circle.copy()
        dashed_circle2.scale(2)

        self.add(pis)
        self.play(ShowCreation(dashed_circle, lag_ratio=0))
        self.play(ShowCreation(dashed_circle2, lag_ratio=0))
        anims = []
        for pi in pis:
            if pi.status == "S":
                anims.append(ApplyMethod(
                    pi.body.change, "pleading", sicky.body.eyes
                ))
        random.shuffle(anims)
        self.play(LaggedStart(*anims))
        self.wait(10)


class CutPInfectionInHalf(Scene):
    def construct(self):
        # Add people
        sicky = PiPerson(
            height=1,
            infection_radius=4,
            max_speed=0,
            wander_step_size=0,
        )
        normy = sicky.deepcopy()
        normy.next_to(sicky, RIGHT)
        normy.body.look_at(sicky.body.eyes)

        circ = Circle(radius=4)
        d_circ = DashedVMobject(circ, num_dashes=30)
        d_circ.set_color(RED)
        d_circ.move_to(sicky)

        sicky.set_status("I")
        self.add(sicky, normy)
        self.add(d_circ)
        self.play(d_circ.scale, 0.5)
        self.wait()

        # Prob label
        eq = VGroup(
            Tex("P(\\text{Infection}) = "),
            DecimalNumber(0.2),
        )
        eq.arrange(RIGHT, buff=0.2)
        eq.to_edge(UP)

        arrow = Vector(0.5 * RIGHT)
        arrow.next_to(eq, RIGHT)
        new_rhs = eq[1].copy()
        new_rhs.next_to(arrow, RIGHT)
        new_rhs.set_color(YELLOW)

        self.play(FadeIn(eq))
        self.play(
            TransformFromCopy(eq[1], new_rhs),
            GrowArrow(arrow)
        )
        self.play(ChangeDecimalToValue(new_rhs, 0.1))
        self.wait(2)

        # Each day
        clock = Clock()
        clock.set_height(1)
        clock.next_to(normy, UR, buff=0.7)

        def get_clock_run(clock):
            return ClockPassesTime(
                clock,
                hours_passed=1,
                run_time=1,
            )

        self.play(
            VFadeIn(clock),
            get_clock_run(clock),
        )

        # Random choice
        choices = VGroup()
        for x in range(9):
            choices.add(Checkmark())
        for x in range(1):
            choices.add(Exmark())
        choices.arrange(DOWN)
        choices.set_height(3)
        choices.next_to(clock, DOWN)

        rect = SurroundingRectangle(choices[0])
        self.add(choices, rect)

        def show_random_choice(scene, rect, choices):
            for x in range(10):
                rect.move_to(random.choice(choices[:-1]))
                scene.wait(0.1)

        show_random_choice(self, rect, choices)

        for x in range(6):
            self.play(get_clock_run(clock))
            show_random_choice(self, rect, choices)
        rect.move_to(choices[-1])
        normy.set_status("I")
        self.add(normy)
        self.play(
            FadeOut(clock),
            FadeOut(choices),
            FadeOut(rect),
        )
        self.wait(4)


class AsymptomaticCases(Scene):
    def construct(self):
        pis = VGroup(*[
            PiPerson(
                height=1,
                infection_radius=2,
                wander_step_size=0,
                max_speed=0,
            )
            for x in range(5)
        ])
        pis.arrange(RIGHT, buff=2)
        pis.to_edge(DOWN, buff=2)

        sneaky = pis[1]
        sneaky.p_symptomatic_on_infection = 0

        self.add(pis)

        for pi in pis:
            if pi is sneaky:
                pi.color_map["I"] = YELLOW
                pi.mode_map["I"] = "coin_flip_1"
            else:
                pi.color_map["I"] = RED
                pi.mode_map["I"] = "sick"
            pi.set_status("I")
            self.wait(0.1)
        self.wait(2)

        label = TexText("Never isolated")
        label.set_height(0.8)
        label.to_edge(UP)
        label.set_color(YELLOW)

        arrow = Arrow(
            label.get_bottom(),
            sneaky.body.get_top(),
            buff=0.5,
            max_tip_length_to_length_ratio=0.5,
            stroke_width=6,
            max_stroke_width_to_length_ratio=10,
        )

        self.play(
            FadeInFromDown(label),
            GrowArrow(arrow),
        )
        self.wait(13)

class IntroduceSocialDistancing(Scene):
    def construct(self):
        pis = VGroup(*[
            PiPerson(
                height=2,
                wander_step_size=0,
                gravity_well=None,
                social_distance_color_threshold=5,
                max_social_distance_stroke_width=10,
                dl_bound=[-FRAME_WIDTH / 2 + 1, -2],
                ur_bound=[FRAME_WIDTH / 2 - 1, 2],
            )
            for x in range(3)
        ])
        pis.arrange(RIGHT, buff=0.25)
        pis.move_to(DOWN)
        pi1, pi2, pi3 = pis

        slider = ValueSlider(
            "Social distance factor",
            0,
            x_min=0,
            x_max=5,
            tick_frequency=1,
            numbers_to_show=range(6),
            marker_color=YELLOW,
        )
        slider.center()
        slider.to_edge(UP)
        self.add(slider)

        def update_pi(pi):
            pi.social_distance_factor = 4 * slider.p2n(slider.marker.get_center())

        for pi in pis:
            pi.add_updater(update_pi)
            pi.repulsion_points = [
                pi2.get_center()
                for pi2 in pis
                if pi2 is not pi
            ]

        self.add(pis)
        self.play(
            FadeIn(slider),
            *[
                ApplyMethod(pi1.body.look_at, pi2.body.eyes)
                for pi1, pi2 in zip(pis, [*pis[1:], pis[0]])
            ]
        )
        self.add(*pis)
        self.wait()
        self.play(slider.get_change_anim(3))
        self.wait(4)

        for i, vect in (0, RIGHT), (2, LEFT):
            pis.suspend_updating()
            pis[1].generate_target()
            pis[1].target.next_to(pis[i], vect, SMALL_BUFF)
            pis[1].target.body.look_at(pis[i].body.eyes)
            self.play(
                MoveToTarget(pis[1]),
                path_arc=PI,
            )
            pis.resume_updating()
            self.wait(5)
        self.wait(5)

        self.embed()


class FastForwardBy2(Scene):
    CONFIG = {
        "n": 2,
    }

    def construct(self):
        n = self.n
        triangles = VGroup(*[
            ArrowTip(start_angle=0)
            for x in range(n)
        ])
        triangles.arrange(RIGHT, buff=0.01)

        label = VGroup(Tex("\\times"), Integer(n))
        label.set_height(0.4)
        label.arrange(RIGHT, buff=SMALL_BUFF)
        label.next_to(triangles, RIGHT, buff=SMALL_BUFF)

        for mob in triangles, label:
            mob.set_color(GREY_A)
            mob.set_stroke(BLACK, 4, background=True)

        self.play(
            LaggedStartMap(
                FadeInFrom, triangles,
                lambda m: (m, 0.4 * LEFT),
            ),
            FadeIn(label, 0.2 * LEFT),
            run_time=1,
        )
        self.play(
            FadeOut(label),
            FadeOut(triangles),
        )


class FastForwardBy4(FastForwardBy2):
    CONFIG = {
        "n": 4,
    }


class QuarteringLines(Scene):
    def construct(self):
        lines = VGroup(
            Line(UP, DOWN),
            Line(LEFT, RIGHT),
        )
        lines.set_width(FRAME_WIDTH)
        lines.set_height(FRAME_HEIGHT, stretch=True)
        lines.set_stroke(WHITE, 3)
        self.play(ShowCreation(lines))
        self.wait()


class Eradicated(Scene):
    def construct(self):
        word = TexText("Eradicated")
        word.set_color(GREEN)
        self.add(word)


class LeftArrow(Scene):
    def construct(self):
        arrow = Vector(2 * LEFT)
        self.play(GrowArrow(arrow))
        self.wait()
        self.play(FadeOut(arrow))


class IndicationArrow(Scene):
    def construct(self):
        vect = Vector(
            0.5 * DR,
            max_tip_length_to_length_ratio=0.4,
            max_stroke_width_to_length_ratio=10,
            stroke_width=5,
        )
        vect.set_color(YELLOW)
        self.play(GrowArrow(vect))
        self.play(FadeOut(vect))


class REq(Scene):
    def construct(self):
        mob = Tex("R_0 = ")[0]
        mob[1].set_color(BLACK)
        mob[2].shift(mob[1].get_width() * LEFT * 0.7)
        self.add(mob)


class DoubleInfectionRadius(Scene):
    CONFIG = {
        "random_seed": 1,
    }

    def construct(self):
        c1 = Circle(radius=0.25, color=RED)
        c2 = Circle(radius=0.5, color=RED)
        arrow = Vector(RIGHT)
        c1.next_to(arrow, LEFT)
        c2.next_to(arrow, RIGHT)

        title = TexText("Double the\\\\infection radius")
        title.next_to(VGroup(c1, c2), UP)

        self.add(c1, title)
        self.wait()
        self.play(
            GrowArrow(arrow),
            TransformFromCopy(c1, c2),
        )
        self.wait()

        c2.label = TexText("4x area")
        c2.label.scale(0.5)
        c2.label.next_to(c2, DOWN)

        for circ, count in (c1, 4), (c2, 16):
            dots = VGroup()
            for x in range(count):
                dot = Dot(color=BLUE)
                dot.set_stroke(BLACK, 2, background=True)
                dot.set_height(0.05)
                vect = rotate_vector(RIGHT, TAU * random.random())
                vect *= 0.9 * random.random() * circ.get_height() / 2
                dot.move_to(circ.get_center() + vect)
                dots.add(dot)
            circ.dot = dots
            anims = [ShowIncreasingSubsets(dots)]
            if hasattr(circ, "label"):
                anims.append(FadeIn(circ.label, 0.5 * UP))
            self.play(*anims)
            self.wait()

class PInfectionSlider(Scene):
    def construct(self):
        slider = ValueSlider(
            "Probability of infection",
            0.2,
            x_min=0,
            x_max=0.2,
            numbers_to_show=np.arange(0.05, 0.25, 0.05),
            decimal_number_config={
                "num_decimal_places": 2,
            },
            tick_frequency=0.05,
        )
        self.add(slider)
        self.wait()
        self.play(slider.get_change_anim(0.1))
        self.wait()
        self.play(slider.get_change_anim(0.05))
        self.wait()


class WideSpreadTesting(Scene):
    def construct(self):
        # Add dots
        dots = VGroup(*[
            DotPerson(
                height=0.2,
                infection_radius=0.6,
                max_speed=0,
                wander_step_size=0,
                p_symptomatic_on_infection=0.8,
            )
            for x in range(600)
        ])
        dots.arrange_in_grid(20, 30)
        dots.set_height(FRAME_HEIGHT - 1)

        self.add(dots)
        sick_dots = VGroup()
        for x in range(36):
            sicky = random.choice(dots)
            sicky.set_status("I")
            sick_dots.add(sicky)
            self.wait(0.1)
        self.wait(2)

        healthy_dots = VGroup()
        for dot in dots:
            if dot.status != "I":
                healthy_dots.add(dot)

        # Show Flash
        rings = self.get_rings(ORIGIN, FRAME_WIDTH + FRAME_HEIGHT, 0.1)
        rings.shift(7 * LEFT)
        for i, ring in enumerate(rings):
            ring.shift(0.05 * i**1.2 * RIGHT)

        self.play(LaggedStartMap(
            FadeIn, rings,
            lag_ratio=3 / len(rings),
            run_time=2.5,
            rate_func=there_and_back,
        ))

        # Quarantine
        box = Square()
        box.set_height(2)
        box.to_corner(DL)
        box.shift(LEFT)

        anims = []
        points = VGroup()
        points_target = VGroup()
        for dot in sick_dots:
            point = VectorizedPoint(dot.get_center())
            point.generate_target()
            points.add(point)
            points_target.add(point.target)

            dot.push_anim(MaintainPositionRelativeTo(dot, point, run_time=3))
            anims.append(MoveToTarget(point))

        points_target.arrange_in_grid()
        points_target.set_width(box.get_width() - 1)
        points_target.move_to(box)

        self.play(
            ShowCreation(box),
            LaggedStartMap(MoveToTarget, points, lag_ratio=0.05),
            self.camera.frame.shift, LEFT,
            run_time=3,
        )
        self.wait(9)

    def get_rings(self, center, max_radius, delta_r):
        radii = np.arange(0, max_radius, delta_r)
        rings = VGroup(*[
            Circle(
                radius=r,
                stroke_opacity=0.75 * (1 - fdiv(r, max_radius)),
                stroke_color=TEAL,
                stroke_width=100 * delta_r,
            )
            for r in radii
        ])
        rings.move_to(center)
        return rings


class VirusSpreading(Scene):
    def construct(self):
        virus = SVGMobject(file_name="virus")
        virus.set_fill(RED_E, 1)
        virus.set_stroke([RED, WHITE], width=0)
        height = 3
        virus.set_height(height)

        self.play(DrawBorderThenFill(virus))

        viruses = VGroup(virus)

        for x in range(8):
            height *= 0.8
            anims = []
            new_viruses = VGroup()
            for virus in viruses:
                children = [virus.copy(), virus.copy()]
                for child in children:
                    child.set_height(height)
                    child.set_color(interpolate_color(
                        RED_E,
                        GREY_D,
                        0.7 * random.random(),
                    ))
                    child.shift([
                        (random.random() - 0.5) * 3,
                        (random.random() - 0.5) * 3,
                        0,
                    ])
                    anims.append(TransformFromCopy(virus, child))
                    new_viruses.add(child)
            new_viruses.center()
            self.remove(viruses)
            self.play(*anims, run_time=0.5)
            viruses.set_submobjects(list(new_viruses))
        self.wait()

        # Eliminate
        for virus in viruses:
            virus.generate_target()
            virus.target.scale(3)
            virus.target.set_color(WHITE)
            virus.target.set_opacity(0)

        self.play(LaggedStartMap(
            MoveToTarget, viruses,
            run_time=8,
            lag_ratio=3 / len(viruses)
        ))
