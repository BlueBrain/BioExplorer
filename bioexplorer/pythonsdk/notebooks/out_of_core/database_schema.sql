create table brick
(
	guid integer not null
		constraint brick_pk
			primary key,
	version integer not null,
	nb_models integer not null,
	buffer bytea not null
);

alter table brick owner to bioexplorer;

create unique index brick_guid_uindex
	on brick (guid);

create table configuration
(
	scene_size_x integer not null,
	scene_size_y integer not null,
	scene_size_z integer not null,
	nb_bricks integer not null,
	description varchar not null
);

alter table configuration owner to bioexplorer;
