create table electrical_type
(
    guid        integer not null
        constraint electrical_type_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table electrical_type
    owner to bioexplorer;

create unique index electrical_type_code_uindex
    on electrical_type (code)
    tablespace bioexplorer_ts;

create unique index electrical_type_guid_uindex
    on electrical_type (guid)
    tablespace bioexplorer_ts;

create table layer
(
    guid        integer not null
        constraint layer_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table layer
    owner to bioexplorer;

create unique index layer_code_uindex
    on layer (code)
    tablespace bioexplorer_ts;

create unique index layer_guid_uindex
    on layer (guid)
    tablespace bioexplorer_ts;

create table morphological_type
(
    guid        integer not null
        constraint morphological_type_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table morphological_type
    owner to bioexplorer;

create unique index morphological_type_code_uindex
    on morphological_type (code)
    tablespace bioexplorer_ts;

create unique index morphological_type_guid_uindex
    on morphological_type (guid)
    tablespace bioexplorer_ts;

create table morphology
(
    guid integer not null
        constraint morphology_pk
            primary key,
    code varchar not null
)
    tablespace bioexplorer_ts;

alter table morphology
    owner to bioexplorer;

create unique index morphology_basename_uindex
    on morphology (code)
    tablespace bioexplorer_ts;

create unique index morphology_guid_uindex
    on morphology (guid)
    tablespace bioexplorer_ts;

create table morphology_class
(
    guid        integer not null
        constraint morphology_class_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table morphology_class
    owner to bioexplorer;

create unique index morphology_class_guid_uindex
    on morphology_class (guid)
    tablespace bioexplorer_ts;

create table region
(
    guid        integer not null
        constraint region_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table region
    owner to bioexplorer;

create unique index region_code_uindex
    on region (code)
    tablespace bioexplorer_ts;

create unique index region_guid_uindex
    on region (guid)
    tablespace bioexplorer_ts;

create table section_type
(
    guid        integer not null
        constraint section_type_pk
            primary key,
    description varchar not null
)
    tablespace bioexplorer_ts;

alter table section_type
    owner to bioexplorer;

create unique index section_type_description_uindex
    on section_type (description)
    tablespace bioexplorer_ts;

create unique index section_type_guid_uindex
    on section_type (guid)
    tablespace bioexplorer_ts;

create table section
(
    morphology_guid     integer                    not null
        constraint section_morphology_guid_fk
            references morphology
            on update cascade on delete cascade,
    section_guid        integer                    not null,
    section_parent_guid integer                    not null,
    section_type_guid   integer                    not null
        constraint section_section_type_guid_fk
            references section_type,
    points              bytea                      not null,
    x                   double precision default 0 not null,
    y                   double precision default 0 not null,
    z                   double precision default 0 not null,
    constraint section_pk
        primary key (morphology_guid, section_guid, section_parent_guid)
)
    tablespace bioexplorer_ts;

alter table section
    owner to bioexplorer;

create index section_morphology_guid_index
    on section (morphology_guid)
    tablespace bioexplorer_ts;

create unique index section_morphology_guid_section_guid_uindex
    on section (morphology_guid, section_guid)
    tablespace bioexplorer_ts;

create table synapse_class
(
    guid        integer not null
        constraint synapse_class_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table synapse_class
    owner to bioexplorer;

create table node
(
    guid                    integer                    not null
        constraint node_pk
            primary key,
    x                       double precision           not null,
    y                       double precision           not null,
    z                       double precision           not null,
    rotation_x              double precision default 0 not null,
    rotation_y              double precision default 0 not null,
    rotation_z              double precision default 0 not null,
    rotation_w              double precision default 1 not null,
    morphology_guid         integer                    not null
        constraint node_morphology_guid_fk
            references morphology
            on update cascade on delete cascade,
    morphology_class_guid   integer                    not null
        constraint node_morphology_class_guid_fk
            references morphology_class
            on update cascade on delete cascade,
    electrical_type_guid    integer                    not null
        constraint node_electrical_type_guid_fk
            references electrical_type
            on update cascade on delete cascade,
    morphological_type_guid integer                    not null
        constraint node_morphological_type_guid_fk
            references morphological_type
            on update cascade on delete cascade
        constraint node_morphological_type_guid_fk_2
            references morphological_type
            on update cascade on delete cascade,
    region_guid             integer                    not null
        constraint node_region_guid_fk
            references region
            on update cascade on delete cascade,
    layer_guid              integer                    not null
        constraint node_layer_guid_fk
            references layer
            on update cascade on delete cascade,
    synapse_class_guid      integer                    not null
        constraint node_synapse_class_guid_fk
            references synapse_class
            on update cascade on delete cascade
)
    tablespace bioexplorer_ts;

alter table node
    owner to bioexplorer;

create unique index node_guid_uindex
    on node (guid)
    tablespace bioexplorer_ts;

create index node_electrical_type_guid_index
    on node (electrical_type_guid)
    tablespace bioexplorer_ts;

create index node_morphological_type_guid_index
    on node (morphological_type_guid)
    tablespace bioexplorer_ts;

create table synapse
(
    guid                     bigint           not null
        constraint synapse_pk
            primary key,
    presynaptic_neuron_guid  integer          not null
        constraint synapse_node_guid_fk
            references node
            on update cascade on delete cascade,
    postsynaptic_neuron_guid integer          not null
        constraint synapse_node_guid_fk_2
            references node
            on update cascade on delete cascade,
    surface_x_position       double precision not null,
    surface_y_position       double precision not null,
    surface_z_position       double precision not null,
    center_x_position        double precision not null,
    center_y_position        double precision not null,
    center_z_position        double precision not null
)
    tablespace bioexplorer_ts;

alter table synapse
    owner to bioexplorer;

create index synapse_presynaptic_neuron_guid_index
    on synapse (presynaptic_neuron_guid)
    tablespace bioexplorer_ts;

create unique index synapse_guid_uindex
    on synapse (guid)
    tablespace bioexplorer_ts;

create unique index synapse_class_guid_uindex
    on synapse_class (guid)
    tablespace bioexplorer_ts;

create table target
(
    guid        integer not null
        constraint target_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table target
    owner to bioexplorer;

create unique index target_code_uindex
    on target (code)
    tablespace bioexplorer_ts;

create unique index target_guid_uindex
    on target (guid)
    tablespace bioexplorer_ts;

create table target_node
(
    node_guid   integer not null
        constraint target_node_node_guid_fk
            references node
            on update cascade on delete cascade,
    target_guid integer not null
        constraint target_node_target_guid_fk
            references target
            on update cascade on delete cascade,
    constraint target_node_pk
        primary key (node_guid, target_guid)
)
    tablespace bioexplorer_ts;

alter table target_node
    owner to bioexplorer;

create function length(anyarray) returns double precision
    immutable
    strict
    language sql
as
$$
select sqrt($1[1] * $1[1] + $1[2] * $1[2] + $1[3] * $1[3]);
$$;

alter function length(anyarray) owner to bioexplorer;

create function norm(anyarray) returns anyarray
    immutable
    strict
    language sql
as
$$
select array[$1[1] / length($1), $1[2] / length($1), $1[3] / length($1)];
$$;

alter function norm(anyarray) owner to bioexplorer;

create function "cross"(anyarray, anyarray) returns anyarray
    immutable
    strict
    language sql
as
$$
select array[$1[2] * $2[3] - $1[3] * $2[2], $1[3] * $2[1] - $1[1] * $2[3], $1[1] * $2[2] - $1[2] * $2[1]];
$$;

alter function "cross"(anyarray, anyarray) owner to bioexplorer;

